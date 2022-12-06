import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

import gym

# logging python
import logging
import sys

# monitoring/logging ML
import wandb

####################
####################

class Net(nn.Module):
    
    def __init__(self) -> None:
        super(Net, self).__init__()

class ValueNet(Net):

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
        return self.layer3(x)
    
    def loss(self, states, returns):
        pass

class PolicyNet(Net):

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
        return self.layer3(x)
    
    def loss(self, states, actions, advantages):
        pass


####################
####################


class PPOAgent:

    def __init__(self, 
        env, 
        in_dim, 
        out_dim, 
        seed=42, 
        gamma=0.99, 
        lr=1e-3) -> None:

        # TODO: Fix hyperparameter
        self.env = env
        self.gamma = gamma
        self.lr = lr
        # seed torch, numpy and gym
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # add net for actor and critic
        self.policyNet = PolicyNet(in_dim, out_dim)
        self.valueNet = ValueNet(in_dim, 1)

        # add optimizer
        self.policyNet_optim = Adam(self.policyNet.parameters(), lr=self.lr)
        self.valueNet_optim = Adam(self.valueNet.parameters(), lr=self.lr)

    def step(self):
        pass

    def policy(self):
        pass

    def finish_episode(self):
        pass

    def collect_rollout(self, state, n_step=1):
        rollout, done = [], False
        for _ in range(n_step): 
            pass

    def train(self):
        pass

    def learn(self):
        pass


####################
####################

def make_env(gym_id='Pendulum-v1'):
    return gym.make(gym_id)

def train():
    # TODO Add Checkpoints to load model 
    pass

def test():
    pass

if __name__ == '__main__':
    
    # Hyperparameter
    unity_file_name = ''
    num_total_steps = 25e3
    learning_rate = 1e-3
    epsilon = 0.2
    max_trajectory_size = 10000
    env_name = 'Pendulum-v1'

    # configure logger
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    env = make_env(env_name)
    # get dimensions of observations (what goes in?)
    # and actions (what goes out?)
    obs_dim = env.observation_space.shape[0] 
    act_dim = env.action_space.shape[0]

    logging.info(f'env observation dim: {obs_dim}')
    logging.info(f'env action dim: {act_dim}')

    agent = PPOAgent(env, in_dim=obs_dim, out_dim=act_dim, lr=learning_rate)
