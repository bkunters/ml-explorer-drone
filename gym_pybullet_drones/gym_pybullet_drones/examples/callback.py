import logging
import sys
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import wandb

# config logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class TrainingCallback(BaseCallback):
    
    def __init__(self, n_rollout_steps=2048, wandb=None, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.n_rollout_steps = n_rollout_steps
        self.steps = 0
        self.over_flag = False
        self.n_ep_rewards = []
        self.wandb = wandb

    def _on_step(self) -> bool:
        """ This method will be called by the model after each call to `env.step()`.
            :return: (bool) If the callback returns False, training is aborted early.
        """

        if(self.locals.get('n_steps')!= self.n_rollout_steps-1):
            if(self.locals.get('dones')):
                self.n_ep_rewards.append(self.locals.get('infos')[0].get('episode').get('r'))
                
                if(self.over_flag):
                    n_ep_rewards = np.array(self.n_ep_rewards)
                    ep_mean_rew = np.mean(n_ep_rewards)
                    ep_min_rew = np.min(n_ep_rewards)
                    ep_max_rew = np.max(n_ep_rewards)

                    logging.info(f'------------ Episode: {self.steps} --------------')
                    logging.info(f"Max ep_return:        {ep_max_rew}")
                    logging.info(f"Min ep_return:        {ep_min_rew}")
                    logging.info(f"Mean ep_return:       {ep_mean_rew}")
                    logging.info('--------------------------------------------')
                    logging.info('\n')

                    if self.wandb:
                        wandb.log({
                        'train/mean episode returns': ep_mean_rew,
                        'train/min episode returns': ep_min_rew,
                        'train/max episode returns': ep_max_rew,

                        }, step=self.steps)
                    self.n_ep_rewards = []
                    self.over_flag = False
                    
        elif(self.locals.get('dones')):
            self.n_ep_rewards.append(self.locals.get('infos')[0].get('episode').get('r'))
            n_ep_rewards = np.array(self.n_ep_rewards)
            ep_mean_rew = np.mean(n_ep_rewards)
            ep_min_rew = np.min(n_ep_rewards)
            ep_max_rew = np.max(n_ep_rewards)

            logging.info(f'------------ Episode: {self.steps} --------------')
            logging.info(f"Max ep_return:        {ep_max_rew}")
            logging.info(f"Min ep_return:        {ep_min_rew}")
            logging.info(f"Mean ep_return:       {ep_mean_rew}")
            logging.info('--------------------------------------------')
            logging.info('\n')

            if self.wandb:
                wandb.log({
                    'train/mean episode returns': ep_mean_rew,
                    'train/min episode returns': ep_min_rew,
                    'train/max episode returns': ep_max_rew,
                    }, step=self.steps)
            
            self.n_ep_rewards = []

        else:
            self.over_flag = True

        self.steps += 1
        return True