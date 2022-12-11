import torch
from torch import nn
from torch.optim import Adam
from  torch.distributions import multivariate_normal
from torch.distributions import Categorical
import numpy as np
import datetime
import gym
import os

# logging python
import logging
import sys

# monitoring/logging ML
import wandb

MODEL_PATH = './models/'
####################
####### TODO #######
####################

# This is a TODO Section - please mark a todo as (done) if done
# 0) Check current implementation against article: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
# 2) Check code for continuous domain --> test with Pendulum
# 4) Fix calculation of Advantage
# 5) Check implementation of cummulative rewards

####################
####################

class Net(nn.Module):
    
    def __init__(self) -> None:
        super(Net, self).__init__()

class ValueNet(Net):
    """Setup Value Network (Critic) optimizer"""
    def __init__(self, in_dim, out_dim) -> None:
        super(ValueNet, self).__init__()
        self.flatten=nn.Flatten()
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        x = self.relu(self.layer1(obs))
        x = self.relu(self.layer2(x))
        out = self.layer3(x)
        return out
    
    def loss(self, obs, rewards):
        """Objective function defined by mean-squared error"""
        values = self(obs).squeeze()
        return ((rewards - values)**2).mean() # MSE loss regression

class PolicyNet(Net):
    """Setup Policy Network (Actor)"""
    def __init__(self, in_dim, out_dim) -> None:
        super(PolicyNet, self).__init__()
        self.flatten=nn.Flatten()
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)
        self.relu = nn.ReLU()

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        x = self.relu(self.layer1(obs))
        x = self.relu(self.layer2(x))
        out = self.layer3(x)
        return out
    
    def loss(self, advantages, a_log_probs, v_log_probs, clip_eps=0.2):
        """Make the clipped objective function to compute loss."""
        ratio = torch.exp(v_log_probs - a_log_probs) # ratio between pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
        clip_1 = ratio * advantages
        clip_2 = torch.clamp(ratio, min=1.0 - clip_eps, max=1.0 + clip_eps) * advantages
        policy_loss = (-torch.min(clip_1, clip_2)).mean() # negative as Adam mins loss - we want to max it
        return policy_loss


####################
####################


class PPO_PolicyGradient:

    def __init__(self, 
        env, 
        in_dim, 
        out_dim,
        total_timesteps,
        max_trajectory_size,
        trajectory_iterations,
        num_epochs=5,
        lr_p=1e-3,
        lr_v=1e-3,
        gamma=0.99,
        epsilon=0.22,
        seed=42) -> None:
        
        # hyperparams
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.total_timesteps = total_timesteps
        self.max_trajectory_size = max_trajectory_size
        self.trajectory_iterations = trajectory_iterations
        self.num_epochs = num_epochs
        self.lr_p = lr_p
        self.lr_v = lr_v
        self.gamma = gamma
        self.epsilon = epsilon

        # env + seeding
        self.env = env
        self.seed = seed

        # seed torch, numpy and gym
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # add net for actor and critic
        self.policy_net = PolicyNet(self.in_dim, self.out_dim) # Setup Policy Network (Actor)
        self.value_net = ValueNet(self.in_dim, 1) # Setup Value Network (Critic)

        # add optimizer for actor and critic
        self.policy_net_optim = Adam(self.policy_net.parameters(), lr=self.lr_p) # Setup Policy Network (Actor) optimizer
        self.value_net_optim = Adam(self.value_net.parameters(), lr=self.lr_v)  # Setup Value Network (Critic) optimizer

    def get_discrete_policy(self, obs):
        """Make function to compute action distribution in discrete action space."""
        # 2) Use Categorial distribution for discrete space
        # https://pytorch.org/docs/stable/distributions.html
        action_prob = self.policy_net.forward(obs) # query Policy Network (Actor) for mean action
        return Categorical(logits=action_prob)

    def get_continuous_policy(self, obs):
        """Make function to compute action distribution in continuous action space."""
        # Multivariate Normal Distribution Lecture 15.7 (Andrew Ng) https://www.youtube.com/watch?v=JjB58InuTqM
        # fixes the detection of outliers, allows to capture correlation between features
        # https://discuss.pytorch.org/t/understanding-log-prob-for-normal-distribution-in-pytorch/73809
        # 1) Use Normal distribution for continuous space
        action_prob = self.policy_net.forward(obs) # query Policy Network (Actor) for mean action
        cov_matrix = torch.diag(torch.full(size=(self.out_dim,), fill_value=0.5))
        return multivariate_normal.MultivariateNormal(action_prob, covariance_matrix=cov_matrix)

    def get_action(self, dist):
        """Make action selection function (outputs actions, sampled from policy)."""
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob
    
    def get_values(self, obs, actions, dist):
        """Make value selection function (outputs values for obs in a batch)."""
        values = self.value_net.forward(obs)
        log_prob = dist.log_prob(actions)
        return values, log_prob

    def step(self, obs):
        """ Given an observation, get action and probabilities from policy network (actor)"""
        action_dist = self.get_continuous_policy(obs) # self.get_discrete_policy(obs)
        action, log_prob = self.get_action(action_dist)
        return action.detach().numpy(), log_prob.detach().numpy()

    def cummulative_reward(self, rewards):
        # Cumulative rewards: https://gongybable.medium.com/reinforcement-learning-introduction-609040c8be36
        # G(t) = R(t) + gamma * R(t-1)
        cum_rewards = []
        for reward in reversed(rewards):
            cumulate_discount = reward + (self.gamma * cumulate_discount)
            cum_rewards.append(cumulate_discount)
        return torch.tensor(np.array(cum_rewards), dtype=torch.float)

    def advantage_estimate(self, rewards, values, normalized=True):
        """Simplest advantage calculation"""
        # STEP 5: compute advantage estimates A_t
        advantages = rewards - values
        if normalized:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        return advantages
    
    def generalized_advantage_estimate(self):
        pass

    def collect_rollout(self, sum_rewards, num_episodes, num_passed_timesteps):
        """Collect a batch of simulated data each time we iterate the actor/critic network (on-policy)"""
    
        trajectory_obs = []
        trajectory_actions = []
        trajectory_rewards = []
        trajectory_action_probs = []
        trajectory_values = []

        next_obs = self.env.reset()
        env.render(mode='human')
        total_reward, num_batches, mean_reward = 0, 0, 0

        logging.info("Collecting batch trajectories...")
        for _ in range(trajectory_iterations):
            while True: 
                # collect observation and get action with log probs
                trajectory_obs.append(next_obs)
                action, log_probability = self.step(next_obs)

                # Run an episode 
                num_batches += 1
                num_passed_timesteps += 1

                action, log_probability = self.step(next_obs)
                
                # STEP 3: collecting set of trajectories D_k by running action 
                # that was sampled from policy in environment
                next_obs, reward, done, _ = self.env.step(action)
                value = self.value_net.forward(next_obs)

                total_reward += reward
                sum_rewards += reward

                # tracking of values
                trajectory_actions.append(action)
                trajectory_action_probs.append(log_probability)
                trajectory_rewards.append(reward)
                trajectory_values.append(value)
                
                # break out of loop if episode is terminated
                if done:
                    next_obs = self.env.reset()
                    # calculate stats and reset all values
                    num_episodes += 1
                    total_reward, trajectory_values = 0, []
                    break
            
        # STEP 4: Calculate rewards to go R_t
        mean_reward = sum_rewards / num_episodes
        logging.info(f"Mean cumulative reward: {mean_reward}")
        
        return torch.tensor(np.array(trajectory_obs), dtype=torch.float), \
                torch.tensor(np.array(trajectory_actions), dtype=torch.float), \
                torch.tensor(np.array(trajectory_action_probs), dtype=torch.float), \
                torch.tensor(np.array(trajectory_rewards), dtype=torch.float), \
                mean_reward, \
                num_passed_timesteps

    def train(self, obs, rewards, advantages, a_log_probs, v_log_probs, clip):
        """Calculate loss and update weights of both networks."""
        self.policy_net_optim.zero_grad() # reset optimizer
        policy_loss = self.policy_net.loss(advantages, a_log_probs, v_log_probs, clip)
        policy_loss.backward()

        self.value_net_optim.zero_grad() # resetself.env.reset() optimizer
        value_loss = self.value_net.loss(obs, rewards)
        value_loss.backward()

        return policy_loss, value_loss

    def learn(self):
        """"""
        # logging info 
        logging.info('Updating the neural network...')
        num_passed_timesteps, mean_reward, sum_rewards, best_mean_reward, num_episodes, t_simulated = 0, 0, 0, 0, 1, 0 # number of timesteps simulated
        while t_simulated < self.total_timesteps:
            policy_loss, value_loss = 0, 0
            # Collect trajectory
            # STEP 3-4: imulate and collect trajectories --> the following values are all per batch
            obs, actions, a_log_probs, rewards2go, mean_reward, num_passed_timesteps = self.collect_rollout(sum_rewards, num_episodes, num_passed_timesteps)
            # reset
            sum_rewards = 0
            # calculate the advantage of current iteration
            values = self.value_net.forward(obs).squeeze()
            # STEP 5: compute advantage estimates A_t
            advantages = self.advantage_estimate(rewards2go, values.detach())

            # loop for network update
            for epoch in range(self.num_epochs):
                # STEP 6-7: calculate loss and update weights
                dist = self.get_continuous_policy(obs) # self.get_discrete_policy(obs)
                values, v_log_probs = self.get_values(obs, actions, dist)
                policy_loss, value_loss = self.train(obs, rewards2go, advantages, a_log_probs, v_log_probs, clip=self.epsilon)
            
            logging.info('###########################################')
            logging.info(f"Epoch: {epoch}, Policy loss: {policy_loss}")
            logging.info(f"Epoch: {epoch}, Value loss: {value_loss}")
            logging.info(f"Total time steps: {num_passed_timesteps}")
            logging.info('###########################################\n')
            
            # logging for monitoring in W&B
            wandb.log({
                'time steps': num_passed_timesteps,
                'policy loss': policy_loss,
                'value loss': value_loss,
                'mean reward': mean_reward})
            
            # store model in checkpoints
            if mean_reward > best_mean_reward:
                env_name = env.unwrapped.spec.id
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.policy_net.state_dict(),
                    'optimizer_state_dict': self.policy_net_optim.state_dict(),
                    'loss': policy_loss,
                    }, f'{MODEL_PATH}{env_name}__policyNet')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.value_net.state_dict(),
                    'optimizer_state_dict': self.value_net_optim.state_dict(),
                    'loss': policy_loss,
                    }, f'{MODEL_PATH}{env_name}__valueNet')
                best_mean_reward = mean_reward

####################
####################

def arg_parser():
    pass 

def make_env(env_id='Pendulum-v1', render_mode=False, seed=42):
    # TODO: Needs to be parallized for parallel simulation
    env = gym.make(env_id)
    return env

def train():
    # TODO Add Checkpoints to load model 
    pass

def test():
    pass

if __name__ == '__main__':
    
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    # Hyperparameter
    unity_file_name = ''            # name of unity environment
    total_timesteps = 1000          # Total number of epochs to run the training
    max_trajectory_size = 10000     # max number of trajectory samples to be sampled per time step. 
    trajectory_iterations = 10      # number of batches of episodes
    num_epochs = 5                  # Number of epochs per time step to optimize the neural networks
    learning_rate_p = 1e-3          # learning rate for policy network
    learning_rate_v = 1e-3          # learning rate for value network
    gamma = 0.99                    # discount factor
    epsilon = 0.2                   # clipping factor
    env_name = 'Pendulum-v1'       # name of OpenAI gym environment
    #'CartPole-v1' 'Pendulum-v1', 'MountainCar-v0'

    # Configure logger
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    
    env = make_env(env_name, render_mode=True)
    # get dimensions of obs (what goes in?)
    # and actions (what goes out?)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    logging.info(f'env observation space: {obs_shape}')
    logging.info(f'env action space: {act_shape}')
    
    obs_dim = obs_shape[0] 
    act_dim = act_shape[0] # 2 at CartPole

    logging.info(f'env observation dim: {obs_dim}')
    logging.info(f'env action dim: {act_dim}')
    
    # upper and lower bound describing the values our obs can take
    logging.info(f'upper bound for env observation: {env.observation_space.high}')
    logging.info(f'lower bound for env observation: {env.observation_space.low}')
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
   
    # Monitoring with W&B
    wandb.init(
    project=f'drone-mechanics-ppo-OpenAIGym',
    entity='drone-mechanics',
    sync_tensorboard=True,
    config={ # stores hyperparams in job
            'total number of epochs': total_timesteps,
            'max sampled trajectories': max_trajectory_size,
            'batches per episode': trajectory_iterations,
            'number of epochs for update': num_epochs,
            'input layer size': obs_dim,
            'output layer size': act_dim,
            'learning rate (policyNet)': learning_rate_p,
            'learning rate (valueNet)': learning_rate_v,
            'gamma (discount)': gamma,
            'epsilon (clipping)': epsilon    
        },
    name=f"{env_name}__{current_time}",
    # monitor_gym=True,
    save_code=True,
    )

    agent = PPO_PolicyGradient(
                env, 
                in_dim=obs_dim, 
                out_dim=act_dim,
                total_timesteps=total_timesteps,
                max_trajectory_size=max_trajectory_size,
                trajectory_iterations=trajectory_iterations,
                num_epochs=num_epochs,
                lr_p=learning_rate_p,
                lr_v=learning_rate_v,
                gamma=gamma,
                epsilon=epsilon)
    
    # run training
    agent.learn()
    logging.info('### Done ###')
    # cleanup 
    env.close()
    wandb.run.finish() if wandb and wandb.run else None