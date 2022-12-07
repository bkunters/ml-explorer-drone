import torch
from torch import nn
import torch.nn.functional as F
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
# 1) Adjust code for discrete domain --> CartPole
# 2) Check code for continuous domain --> Pendulum
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
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)
    
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        x = F.relu(self.layer1(obs))
        x = F.relu(self.layer2(x))
        out = self.layer3(x)
        return out
    
    def loss(self, obs, rewards):
        """Objective function defined by mean-squared error"""
        return ((rewards - self(obs))**2).mean() # regression

class PolicyNet(Net):
    """Setup Policy Network (Actor)"""
    def __init__(self, in_dim, out_dim) -> None:
        super(PolicyNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, obs, act=None):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        x = F.relu(self.layer1(obs))
        x = F.relu(self.layer2(x))
        out = self.layer3(x)
        return out
    
    def loss(self, obs, actions, advantages, log_probs):
        # TODO: Implement clipped objective function
        # 1. Calculate V_phi and pi_theta(a_t | s_t)
        # 2. Calculate ratio between pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
        # 3. ... 
        pass


####################
####################


class PPO_PolicyGradient:

    def __init__(self, 
        env,
        in_dim, 
        out_dim,
        total_timesteps=10000,
        timesteps_per_batch=2048, # timesteps per batch (number of actions taken in the environments)
        max_timesteps_per_episode=1600, # (number of actions taken in the environments)
        n_updates_per_iteration=5,
        clip=0.2,
        gamma=0.99, 
        lr=1e-3,
        seed=42) -> None:

        # TODO: Check these values --> maybe simplify
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.clip = clip
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.total_timesteps = total_timesteps
        self.timesteps_per_batch = timesteps_per_batch
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.n_update_per_iteration = n_updates_per_iteration

        # seed torch, numpy and gym
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # TODO: Move this to the network defintion
        # add net for actor and critic
        self.policyNet = PolicyNet(self.in_dim, self.out_dim) # Setup Policy Network (Actor)
        self.valueNet = ValueNet(self.in_dim, 1) # Setup Value Network (Critic)

        # add optimizer for actor and critic
        self.policyNet_optim = Adam(self.policyNet.parameters(), lr=self.lr) # Setup Policy Network (Actor) optimizer
        self.valueNet_optim = Adam(self.valueNet.parameters(), lr=self.lr)  # Setup Value Network (Critic) optimizer

    def get_discrete_action_dist(self, obs):
        """Make function to compute action distribution in discrete action space."""
        # 2) Use Categorial distribution for discrete space
        # https://pytorch.org/docs/stable/distributions.html
        action_prob = self.policyNet.forward(obs) # query Policy Network (Actor) for mean action
        return Categorical(logits=action_prob)


    def get_continuous_action_dist(self, obs):
        """Make function to compute action distribution in continuous action space."""
        # Multivariate Normal Distribution Lecture 15.7 (Andrew Ng) https://www.youtube.com/watch?v=JjB58InuTqM
        # fixes the detection of outliers, allows to capture correlation between features
        # https://discuss.pytorch.org/t/understanding-log-prob-for-normal-distribution-in-pytorch/73809
        # 1) Use Normal distribution for continuous space
        action_prob = self.policyNet.forward(obs) # query Policy Network (Actor) for mean action
        cov_matrix = torch.diag(torch.full(size=(self.out_dim,), fill_value=0.5))
        return multivariate_normal.MultivariateNormal(action_prob, covariance_matrix=cov_matrix)

    def get_action(self, dist):
        """Sample a random action from distribution."""
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob
        
    def get_value(self, obs):
        value = self.valueNet.forward(obs)
        return value

    def step(self, obs):
        """ Given an observation, get action and probabilities from policy network (actor)"""
        action_dist = self.get_continuous_action_dist(obs)
        action, log_prob = self.get_action(action_dist)

        # detach and convert to numpy array
        logging.info(f'Sampled action {action}')
        logging.info(f'Sampled probability {log_prob}')
        return action.detach().numpy(), log_prob.detach().numpy()

    def rewards_to_go(self, rewards):
        """Calculate rewards to go to reduce the variance in the policy gradient"""
        # Lecture 5, p. 17: UCB http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf
        # Open AI Documentation: https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#implementing-reward-to-go-policy-gradient
        amount = len(rewards)
        reward_to_go = np.zeros_like(rewards)
        for i in reversed(range(amount)):
            # TODO: Check this function is a discount factor? 
            reward_to_go[i] = rewards[i] + (reward_to_go[i+1] if i+1 < amount else 0)
        return torch.tensor(reward_to_go, dtype=torch.float)

    def advantage(self, rewards, values):
        """Simplest advantage calculation"""
        return rewards - values # TODO: Correct? Eventually normalize advantage (?) if training is very instable

    def collect_rollout(self):
        """Collect a batch of simulated data each time we iterate the actor/critic network (on-policy)"""
        
        logging.info(f'Rollout collecting sample data ...')
        
        episode_lengths_per_batch = [] # lengths of each episode this batch 
        observations_per_batch = [] # observations for this batch - shape (n_timesteps, dim observations)
        actions_per_batch = [] # actions for this batch - shape (n_timesteps, dim actions)
        log_probs_per_batch = [] # log probabilities per action this batch - shape (n_timesteps)
        rewards_per_batch = [] # rewards collected this batch - shape (n_timesteps)

        timesteps_simulated = 0 # number of timesteps simulated
        while timesteps_simulated < self.timesteps_per_batch:

            # track rewards per episode
            rewards_per_episode = []
            # reset environment for new episode
            next_obs = self.env.reset() 
            done = False 

            for episode in range(self.max_timesteps_per_episode):
                # Run an episode 
                timesteps_simulated += 1
                observations_per_batch.appen(next_obs)
                action, log_probability = self.step(next_obs)
                next_obs, reward, done, truncated = self.env.step(action)

                # tracking of values
                actions_per_batch.append(action)
                log_probs_per_batch.append(log_probability)
                rewards_per_episode(reward)
                
                # break out of loop if episode is terminated
                if done or truncated:
                    break
            
            rewards_per_batch(rewards_per_episode)
            episode_lengths_per_batch(episode + 1) # how long was the episode + 1 as we start at 0
            rewards_to_go_per_batch = self.rewards_to_go(rewards_per_batch, dtype=torch.float)

        return torch.tensor(observations_per_batch, dtype=torch.float), \
                torch.tensor(actions_per_batch, dtype=torch.float), \
                torch.tensor(log_probs_per_batch, dtype=torch.float), \
                torch.tensor(rewards_to_go_per_batch), \
                episode_lengths_per_batch

    def train(self, obs, actions, rewards, log_probs, advantages):
        """Calculate loss and update weights of both networks."""
        self.policyNet_optim.zero_grad() # reset optimizer
        policy_loss = self.policyNet.loss(obs, actions, advantages, log_probs)
        policy_loss.backward()

        self.valueNet_optim.zero_grad()
        value_loss = self.valueNet.loss(obs, rewards)
        value_loss.backward()

        # monitoring W&B
        wandb.log({'policy loss': policy_loss, 'value loss': value_loss})

    def learn(self):
        """"""
        # logging info 
        logging.info(f'Updating the network...')
        timesteps_simulated = 0 # number of timesteps simulated
        while timesteps_simulated < self.total_timesteps:
            # simulate and collect trajectories --> the following values are all per batch
            observations, actions, log_probs, rewards2go, episode_length_per_batch = self.collect_rollout()
            # calculate the advantage of current iteration
            values = self.valueNet.forward(observations).detach()
            advantages = self.advantage(rewards2go, values.squeeze())

            for _ in range(self.n_updates_per_iteration):
                # calculate loss and update weights
                self.train(observations, actions, rewards2go, advantages)
            
            # monitoring W&B
            wandb.log({
                'episode length': episode_length_per_batch,
                'mean reward': np.mean(rewards2go.detach().numpy()), 
                'sum rewards': np.sum(rewards2go.detach().numpy())
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
    max_timesteps_per_episode = 1600
    n_updates_per_iteration = 5
    learning_rate = 1e-3
    gamma = 0.99 
    clip = 0.2
    env_name = 'Pendulum-v1' #'CartPole-v1'

    # Configure logger
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    
    env = make_env(env_name)
    # get dimensions of observations (what goes in?)
    # and actions (what goes out?)
    obs_dim = env.observation_space.shape[0] 
    act_dim = env.action_space.shape[0]

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
            'updates per iteration': n_updates_per_iteration,
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