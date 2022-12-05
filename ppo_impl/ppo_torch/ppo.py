import torch
from torch import nn
import numpy as np

####################
####################

class Net(nn.Module):

    def __init__(self) -> None:
        super().__init__()

class ValueNet(Net):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, obs):
        pass
    
    def loss(self, states, returns):
        pass

class PolicyNet(Net):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs, act=None):
        pass
    
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

    def collect_rollout(self):
        pass

    def train(self):
        pass

    def learn(self):
        pass

####################
####################

if __name__ == '__main__':
    pass