import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import wandb
import csv

class TrainingCallback(BaseCallback):
    """
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=1):
        super(TrainingCallback, self).__init__(verbose)
        self.n_ep_rewards = []
        self.max_batch_size = 512
        self.n_rollout_steps = 2048
        
        self.step = 0
        self.t_steps = 0       #steps per rollout

        self.over_flag = False
        self.f_csv = open('tmp/progress_callback.csv', 'w', newline='')
        self.writer = csv.writer(self.f_csv, delimiter=',')
        self.writer.writerow(['step', 'mean_reward', 'max_reward', 'min_reward'])
    def _on_step(self):
        if(self.t_steps!=self.n_rollout_steps-1):
            if(self.locals.get('dones')):
                self.n_ep_rewards.append(self.locals.get('infos')[0].get('episode').get('r'))
                if(self.over_flag):
                    mean_rewards = np.mean(self.n_ep_rewards)
                    max_reward = max(self.n_ep_rewards)
                    min_reward = min(self.n_ep_rewards)
                    print('------', self.step+1, mean_rewards, max_reward, min_reward)
                    self.writer.writerow([self.step+1, mean_rewards, max_reward, min_reward])

                    # wandb.log({
                    #     'mean_episodes_rewards': self.mean_rewards,
                    #
                    # }, step=self.step)
                    self.n_ep_rewards = []
                    self.over_flag = False
                    self.t_steps = 0
        elif(self.locals.get('dones')):
            self.n_ep_rewards.append(self.locals.get('infos')[0].get('episode').get('r'))
            mean_rewards = np.mean(self.n_ep_rewards)
            max_reward = max(self.n_ep_rewards)
            min_reward = min(self.n_ep_rewards)
            print('------', self.step+1, mean_rewards, max_reward, min_reward)
            self.writer.writerow([self.step+1, mean_rewards, max_reward, min_reward])
            # wandb.log({
            #     'mean_episodes_rewards': self.mean_rewards,
            #
            # }, step=self.step)
            self.n_ep_rewards = []
            self.t_steps = 0
        else:
            self.over_flag = True

        # print(self.locals)
        self.f_csv.flush()
        self.step += 1
        self.t_steps += 1
        return True  # returns True, training continues.


tmp_path = "tmp/"
# set up logger
new_logger = configure(tmp_path, ["csv"])

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1_000_000,
    "env_name": "Pendulum-v1",
    "n_steps": 2048,
}

# run = wandb.init(
#     project="log_test_zhengjie",
#     config=config,
#     entity='drone-mechanics',
#     name='stable_baselines3'
# )

model = PPO(config["policy_type"], config["env_name"], n_steps=config["n_steps"], verbose=1)
model.set_logger(new_logger)
model.learn(total_timesteps=config["total_timesteps"], callback=TrainingCallback())

# run.finish()
