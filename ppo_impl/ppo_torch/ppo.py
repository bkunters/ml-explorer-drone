import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

####################
####################

class Net(nn.Module):

    def __init__(self, in_dim, out_dim) -> None:
        super(Net, self).__init__()
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

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

    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs, act=None):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
    
    def loss(self, states, actions, advantages):
        pass

####################
####################


class PPOAgent:

    def __init__(self, env, seed=42, gamma=0.99) -> None:
        self.env = env
        self.gamma = gamma
        # seed torch, numpy and gym
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

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

if __name__ == '__main__':
    pass