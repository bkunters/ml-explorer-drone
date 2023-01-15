import stable_baselines3
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import csv
import wandb

from stable_baselines3.common.logger import configure
import pandas as pd

tmp_path = "tmp/sb3_log/"
# set up logger
new_logger = configure(tmp_path, ["csv"])

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 5_000_000,
    "env_name": "Pendulum-v1",
    "n_steps": 2600,
}

model = PPO(config["policy_type"], config["env_name"], n_steps=config["n_steps"], verbose=1)
model.set_logger(new_logger)
model.learn(total_timesteps=config["total_timesteps"])

run = wandb.init(
    project="log_test_zhengjie",
    config=config,
    entity='drone-mechanics'
)
info = pd.read_csv('tmp/sb3_log/progress.csv')
steps = info['time/total_timesteps']
mean_episode_reward = info['rollout/ep_rew_mean']
ep_lenth = info['rollout/ep_len_mean']
ep_time = info['time/time_elapsed']
for i in range(len(steps)):
    wandb.log({
        'info/timesteps': steps[i],
        'info/episodic returns': mean_episode_reward[i],
        'info/episodic length': ep_lenth[i],
        'info/episodic time': ep_time[i]
        })

run.finish()
