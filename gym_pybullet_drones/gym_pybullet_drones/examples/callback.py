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
        self.stats_drone_data = { 
            "dist_to_start_point": [], 
            "y_position": [], 
            "x_velocity": [], 
            "y_velocity": [], 
            "z_velocity": []
        }
        self.wandb = wandb
        self.iteration_steps = 0

    def _on_step(self) -> bool:
        """ This method will be called by the model after each call to `env.step()`.
            :return: (bool) If the callback returns False, training is aborted early.
        """

        if(self.iteration_steps!= self.n_rollout_steps-1):
            
            if(self.locals.get('dones')):
                info = self.locals.get('infos')
                self.n_ep_rewards.append(info[0].get('episode').get('r'))
                
                self.stats_drone_data['dist_to_start_point'].append(info[0].get('dist_to_start_point'))
                self.stats_drone_data['x_velocity'].append(info[0].get('x_velocity'))
                self.stats_drone_data['y_velocity'].append(info[0].get('y_velocity'))
                self.stats_drone_data['z_velocity'].append(info[0].get('z_velocity'))
                self.stats_drone_data['y_position'].append(info[0].get('y_position'))
                
                if(self.over_flag):
                    
                    dist_origin = np.array(self.stats_drone_data['dist_to_start_point'])
                    x_vel = np.array(self.stats_drone_data['x_velocity'])
                    y_vel = np.array(self.stats_drone_data['y_velocity'])
                    z_vel = np.array(self.stats_drone_data['z_velocity'])
                    y_pos = np.array(self.stats_drone_data['y_position'])

                    n_ep_rewards = np.array(self.n_ep_rewards)
                    ep_mean_rew = np.mean(n_ep_rewards)
                    ep_min_rew = np.min(n_ep_rewards)
                    ep_max_rew = np.max(n_ep_rewards)

                    logging.info(f'------------ Episode: {self.steps} --------')
                    logging.info(f"Max ep_return:        {ep_max_rew}")
                    logging.info(f"Min ep_return:        {ep_min_rew}")
                    logging.info(f"Mean ep_return:       {ep_mean_rew}")
                    logging.info('--------------------------------------------')
                    logging.info('\n')

                    if self.wandb:
                        wandb.log({
                            'drone/dist to start': np.mean(dist_origin),
                            'drone/y position': np.mean(y_pos),
                            'drone/x velocity': np.mean(x_vel),
                            'drone/y velocity': np.mean(y_vel),
                            'drone/z velocity': np.mean(z_vel),
                            'train/mean episode returns': ep_mean_rew,
                            'train/min episode returns': ep_min_rew,
                            'train/max episode returns': ep_max_rew,
                            }, step=self.steps)
                    self.n_ep_rewards = []
                    self.over_flag = False
                    self.iteration_steps = 0
                    if self.stats_drone_data:
                        # remove old values
                        for value in self.stats_drone_data.values():
                            del value[:]

                    
        elif(self.locals.get('dones')):

            info = self.locals.get('infos')
            self.n_ep_rewards.append(info[0].get('episode').get('r'))

            self.stats_drone_data['dist_to_start_point'].append(info[0].get('dist_to_start_point'))
            self.stats_drone_data['x_velocity'].append(info[0].get('x_velocity'))
            self.stats_drone_data['y_velocity'].append(info[0].get('y_velocity'))
            self.stats_drone_data['z_velocity'].append(info[0].get('z_velocity'))
            self.stats_drone_data['y_position'].append(info[0].get('y_position'))
            
            dist_origin = np.array(self.stats_drone_data['dist_to_start_point'])
            x_vel = np.array(self.stats_drone_data['x_velocity'])
            y_vel = np.array(self.stats_drone_data['y_velocity'])
            z_vel = np.array(self.stats_drone_data['z_velocity'])
            y_pos = np.array(self.stats_drone_data['y_position'])

            n_ep_rewards = np.array(self.n_ep_rewards)
            ep_mean_rew = np.mean(n_ep_rewards)
            ep_min_rew = np.min(n_ep_rewards)
            ep_max_rew = np.max(n_ep_rewards)

            logging.info(f'------------ Episode: {self.steps} --------')
            logging.info(f"Max ep_return:        {ep_max_rew}")
            logging.info(f"Min ep_return:        {ep_min_rew}")
            logging.info(f"Mean ep_return:       {ep_mean_rew}")
            logging.info('--------------------------------------------')
            logging.info('\n')

            if self.wandb:
                wandb.log({
                    'drone/dist to start': np.mean(dist_origin),
                    'drone/y position': np.mean(y_pos),
                    'drone/x velocity': np.mean(x_vel),
                    'drone/y velocity': np.mean(y_vel),
                    'drone/z velocity': np.mean(z_vel),
                    'train/mean episode returns': ep_mean_rew,
                    'train/min episode returns': ep_min_rew,
                    'train/max episode returns': ep_max_rew,
                    }, step=self.steps)
            
            self.n_ep_rewards = []
            self.iteration_steps = 0
            if self.stats_drone_data:
                # remove old values
                for value in self.stats_drone_data.values():
                    del value[:]

        else:
            self.over_flag = True

        self.steps += 1
        self.iteration_steps += 1
        return True
