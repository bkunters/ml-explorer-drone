import torch
from torch import nn
from torch.optim import Adam
from  torch.distributions import multivariate_normal
from torch.distributions import Categorical
import numpy as np
import datetime
import gym

# logging python
import logging
import sys

# monitoring/logging ML
import wandb


####################
####### TODO #######
####################

# This is a TODO Section - please mark a todo as (done) if done
# 0) Check current implementation against article: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
# 1) Check code for discrete domain --> test with CartPole (adjust beforehand)
# 2) Check code for continuous domain --> test with Pendulum
# 3) Implement Surrogate clipping loss
# 4) Fix calculation of Advantage
# 5) Check hyperparameters --> check overall implementation logic
# 6) Add checkpoints to restart model if it got interrupted
# 7) Check monitoring on W&B --> eventually we need to change the values logged

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
        return nn.MSELoss()(self(obs).squeeze(), rewards) # regression

class PolicyNet(Net):
    """Setup Policy Network (Actor)"""
    def __init__(self, in_dim, out_dim) -> None:
        super(PolicyNet, self).__init__()
        self.flatten=nn.Flatten()
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        x = self.tanh(self.layer1(obs))
        x = self.tanh(self.layer2(x))
        out = self.softmax(self.layer3(x))
        return out
    
    def loss(self, advantages, action_log_probs, v_log_probs, clip_eps=0.2):
        """Make the clipped objective function to compute loss."""
        ratio = torch.exp(v_log_probs - action_log_probs) # ratio between pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
        clip_1 = ratio * advantages
        clip_2 = torch.clamp(ratio, min=1.0 - clip_eps, max=1.0 + clip_eps) * advantages
        policy_loss = (-torch.min(clip_1, clip_2)).mean()
        return policy_loss


####################
####################


class PPO_PolicyGradient:

    def __init__(self, # TODO: Change hyperparams
        env,
        in_dim, 
        out_dim,
        total_timesteps=10000,
        timesteps_per_batch=2048, # timesteps per batch (number of actions taken in the environments)
        timesteps_per_episode=1600, # timesteps per episode
        updates=5,
        clip=0.2,
        gamma=0.99, 
        lr=1e-3,
        seed=42) -> None:

        self.env = env
        self.seed = seed
        self.gamma = gamma
        self.lr = lr
        self.clip = clip
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.total_timesteps = total_timesteps
        self.timesteps_per_batch = timesteps_per_batch
        self.timesteps_per_episode = timesteps_per_episode
        self.updates = updates

        # seed torch, numpy and gym
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # TODO: Move this to the network defintion
        # add net for actor and critic
        self.policy_net = PolicyNet(self.in_dim, self.out_dim) # Setup Policy Network (Actor)
        self.value_net = ValueNet(self.in_dim, 1) # Setup Value Network (Critic)

        # add optimizer for actor and critic
        self.policyNet_optim = Adam(self.policy_net.parameters(), lr=self.lr) # Setup Policy Network (Actor) optimizer
        self.value_net_optim = Adam(self.value_net.parameters(), lr=self.lr)  # Setup Value Network (Critic) optimizer

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
        """Make value selection function (outputs values for observations in a batch)."""
        values = self.value_net.forward(obs)
        log_prob = dist.log_prob(actions)
        return values, log_prob

    def step(self, obs):
        """ Given an observation, get action and probabilities from policy network (actor)"""
        action_dist = self.get_discrete_policy(obs)
        action, log_prob = self.get_action(action_dist)
        return action.detach().numpy(), log_prob.detach().numpy()

    def reward_to_go(self):
        pass

    def cummulative_reward(self, rewards):
        # Cumulative rewards: https://gongybable.medium.com/reinforcement-learning-introduction-609040c8be36
        # G(t) = R(t) + gamma * R(t-1)
        cum_rewards = []
        for ep_rewards in reversed(rewards):
            cumulate_discount = 0
            for reward in reversed(ep_rewards):
                cumulate_discount = reward + (self.gamma * cumulate_discount)
                cum_rewards.append(cumulate_discount)
        return torch.tensor(np.array(cum_rewards))

    def advantage_estimate(self, rewards, values):
        """Simplest advantage calculation"""
        # STEP 5: compute advantage estimates A_t
        # TODO: eventually normalize advantage if training is instable
        return rewards - values 
    
    def generalized_advantage_estimate(self):
        pass

    def collect_rollout(self):
        """Collect a batch of simulated data each time we iterate the actor/critic network (on-policy)"""
        episode_lengths_per_batch = [] # lengths of each episode this batch 
        observations_per_batch = [] # observations for this batch - shape (n_timesteps, dim observations)
        actions_per_batch = [] # actions for this batch - shape (n_timesteps, dim actions)
        log_probs_per_batch = [] # log probabilities per action this batch - shape (n_timesteps)
        rewards_per_batch = [] # rewards collected this batch - shape (n_timesteps)

        t_simulated = 0 # number of timesteps simulated
        while t_simulated < self.timesteps_per_batch:

            # track rewards per episode
            rewards_per_episode = []
            # reset environment for new episode
            next_obs = self.env.reset() 
            done = False 

            for episode in range(self.timesteps_per_episode):
                # Run an episode 
                t_simulated += 1
                observations_per_batch.append(next_obs)
                action, log_probability = self.step(next_obs)
                
                # STEP 3: collecting set of trajectories D_k by running action 
                # that was sampled from policy in environment
                next_obs, reward, done, truncated = self.env.step(action)

                # tracking of values
                actions_per_batch.append(action)
                log_probs_per_batch.append(log_probability)
                rewards_per_episode.append(reward)
                
                # break out of loop if episode is terminated
                if done or truncated:
                    break
            
            rewards_per_batch.append(rewards_per_episode)
            episode_lengths_per_batch.append(episode + 1) # how long was the episode + 1 as we start at 0
            # STEP 4: Calculate rewards to go R_t
            rewards_to_go_per_batch = self.cummulative_reward(rewards_per_batch)

        return torch.tensor(np.array(observations_per_batch), dtype=torch.float), \
                torch.tensor(np.array(actions_per_batch), dtype=torch.float), \
                torch.tensor(np.array(log_probs_per_batch), dtype=torch.float), \
                rewards_to_go_per_batch, episode_lengths_per_batch

    def train(self, obs, rewards, advantages, action_log_probs, v_log_probs, clip):
        """Calculate loss and update weights of both networks."""
        self.policyNet_optim.zero_grad() # reset optimizer
        policy_loss = self.policy_net.loss(advantages, action_log_probs, v_log_probs, clip)
        policy_loss.backward()

        self.value_net_optim.zero_grad() # reset optimizer
        value_loss = self.value_net.loss(obs, rewards)
        value_loss.backward()

        # logging for monitoring in W&B
        wandb.log({'policy loss': policy_loss, 'value loss': value_loss})

    def learn(self):
        """"""
        # logging info 
        logging.info('Updating the neural network...')
        t_simulated = 0 # number of timesteps simulated
        while t_simulated < self.total_timesteps:
            # STEP 3-4: imulate and collect trajectories --> the following values are all per batch
            observations, actions, action_log_probs, rewards2go, episode_length_per_batch = self.collect_rollout()
            # calculate the advantage of current iteration
            values = self.value_net.forward(observations).squeeze()
            # STEP 5: compute advantage estimates A_t
            advantages = self.advantage_estimate(rewards2go, values.detach())

            for _ in range(self.updates):
                # STEP 6-7: calculate loss and update weights
                dist = self.get_discrete_policy(observations)
                values, log_probs = self.get_values(observations, actions, dist)
                self.train(observations, rewards2go, advantages, action_log_probs, log_probs, clip=self.clip)
            
            # monitoring W&B
            wandb.log({
                'episode length': episode_length_per_batch,
                'mean reward': np.mean(rewards2go.detach().numpy()), # TODO: Check values here correct
                'sum rewards': np.sum(rewards2go.detach().numpy()) # TODO: Check values here correct
                })

####################
####################

def arg_parser():
    pass 

def make_env(env_id='Pendulum-v1', seed=42):
    # TODO: Needs to be parallized for parallel simulation
    env = gym.make(env_id)
    return env

def train():
    # TODO Add Checkpoints to load model 
    pass

def test():
    pass

if __name__ == '__main__':
    
    # Hyperparameter
    unity_file_name = ''
    total_timesteps = 1000
    timesteps_per_batch = 4800
    timesteps_per_episode = 1600
    updates = 5
    learning_rate = 1e-3
    gamma = 0.99 
    clip = 0.2
    env_name = 'CartPole-v1' #'CartPole-v1' 'Pendulum-v1', 'MountainCar-v0'

    # Configure logger
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    
    env = make_env(env_name)
    # get dimensions of observations (what goes in?)
    # and actions (what goes out?)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    logging.info(f'env observation space: {obs_shape}')
    logging.info(f'env action space: {act_shape}')
    
    obs_dim = obs_shape[0] 
    act_dim = 2 # act_shape[0]

    logging.info(f'env observation dim: {obs_dim}')
    logging.info(f'env action dim: {act_dim}')
    
    # upper and lower bound describing the values our observations can take
    logging.info(f'upper bound for env observation: {env.observation_space.high}')
    logging.info(f'lower bound for env observation: {env.observation_space.low}')
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
   
    # Monitoring with W&B
    wandb.init(
    project=f'drone-mechanics-ppo',
    entity='drone-mechanics',
    sync_tensorboard=True,
    config={ # stores hyperparams in job
            'timesteps per batch': timesteps_per_batch,
            'updates per iteration': updates,
            'input layer size': obs_dim,
            'output layer size': act_dim,
            'learning rate': learning_rate,
            'gamma': gamma
    },
    name=f"{env_name}__{current_time}",
    # monitor_gym=True,
    save_code=True,
    )

    agent = PPO_PolicyGradient(env, in_dim=obs_dim, out_dim=act_dim, \
            total_timesteps=total_timesteps,
            timesteps_per_batch=timesteps_per_batch, \
            gamma=gamma, clip=clip, lr=learning_rate)
    
    # run training
    agent.learn()
    logging.info('Done')
    # cleanup 
    env.close()
    wandb.run.finish() if wandb and wandb.run else None