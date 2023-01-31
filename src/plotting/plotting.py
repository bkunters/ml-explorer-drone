import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import seaborn as sns
import os


def concat_csv_files(folder_path):
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    df_list = []
    for file in all_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df_list.append(df)
    return pd.concat(df_list, axis=0, ignore_index=True)

def plot_seaborn_fill(df, x_val, y_val, title='title', smoothing=2, wandb=None, color=None):      
    # Calculate the mean and standard deviation of the mean episodic reward
    mean = df.groupby(x_val).mean()[y_val]
    std = df.groupby(x_val).std()[y_val]

    # Apply Gaussian smoothing
    std_smoothed = gaussian_filter1d(std, smoothing)
    mean_smoothed = gaussian_filter1d(mean, smoothing)

    # random colour generator
    if not color:
        color = np.random.random(), np.random.random(), np.random.random()
    else:
        color = sns.xkcd_rgb[color]

    # Plot the mean and standard deviation as a fill between
    ax = sns.lineplot(x=mean.index, y=mean_smoothed, color=color, label=title)
    ax.fill_between(mean.index, mean - std_smoothed, mean + std_smoothed, color=color, alpha=0.2)

##### Load the csv data into pandas dataframes #####
# sb3
df_ppo_sb3_runs = concat_csv_files("ppo-sb3-runs")

# random agent
df_random_runs = concat_csv_files("random-runs")

# basic hyperparam 
# names: gae, ac, TDac, ra
# (reinforce advantage, normalize ret/adv)
# (TD actor-critic advantage, normalize ret)
# (seperate network, actor-critic advantage, normalize adv)
# (seperate network, GAE advantage, normalize ret)
df_ppo_ra_basic_runs = concat_csv_files("ppo-ra-basic-runs")
df_ppo_tdac_basic_runs = concat_csv_files("ppo-tdac-basic-runs")
df_ppo_ac_basic_runs = concat_csv_files("ppo-ac-basic-runs")
df_ppo_gae_basic_runs = concat_csv_files("ppo-gae-basic-runs")

# advanced hyperparam
# names: gae, ac, TDac, ra
# (GAE advantage, new hyperparams, normalize ret)
# (actor-critic advantage, new hyperparams, normalize adv)
df_ppo_gae_adv_runs = concat_csv_files("ppo-gae-adv-runs")
df_ppo_ac_adv_runs = concat_csv_files("ppo-ac-adv-runs")

# plots
"""
# basic hyperparam
plot_seaborn_fill(df_ppo_ra_basic_runs, x_val='timestep', y_val='mean episodic returns', 
                    title='Our PPO (reinforce advantage, normalize ret/adv)',
                    smoothing=2)

plot_seaborn_fill(df_ppo_tdac_basic_runs, x_val='timestep', y_val='mean episodic returns', 
                    title='Our PPO (TD actor-critic advantage, normalize ret)',
                    smoothing=2)

plot_seaborn_fill(df_ppo_ac_basic_runs, x_val='timestep', y_val='mean episodic returns', 
                    title='Our PPO (seperate network, actor-critic advantage, normalize adv)',
                    smoothing=2)

plot_seaborn_fill(df_ppo_gae_basic_runs, x_val='timestep', y_val='mean episodic returns', 
                    title='Our PPO (seperate network, GAE advantage, normalize ret)',
                    smoothing=2)

"""

# adv hyperparam
plot_seaborn_fill(df_ppo_gae_adv_runs, x_val='timestep', y_val='mean episodic returns', 
                    title='Our PPO (GAE advantage, new hyperparams, normalize ret)',
                    smoothing=2)

plot_seaborn_fill(df_ppo_ac_adv_runs, x_val='timestep', y_val='mean episodic returns', 
                    title='Our PPO (actor-critic advantage, new hyperparams, normalize adv)',
                    smoothing=2)

# sb3
plot_seaborn_fill(df_ppo_sb3_runs, x_val='step', y_val='mean_reward', 
                    title='Stable baseline PPO',
                    smoothing=2)

# random
plot_seaborn_fill(df_random_runs, x_val='Step', y_val='mean_episodes_rewards', 
                    title='Random agent',
                    smoothing=2)

# param plot Pendulum-v1
plt.legend(loc='lower right')
plt.title('Pendulum-v1')
plt.xlabel('Steps')
plt.ylabel('Mean Episodic Return')
plt.grid(visible=True)
plt.ticklabel_format(style='plain', axis='x')
plt.xlim(left = 0, right = 1_000_000)

'''
##### takeoff #####
##### Load the csv data into pandas dataframes #####
# ppo_v2
df_takeoff_ppo_v2_runs = concat_csv_files("takeoff-ppo-v2-runs")

# sb3
df_takeoff_ppo_sb3_runs = concat_csv_files("takeoff-sb3-runs")

# plots
plot_seaborn_fill(df_takeoff_ppo_v2_runs, x_val='timestep', y_val='mean episodic returns', 
                    title='Our PPO (GAE advantage, new hyperparams, normalize ret)',
                    smoothing=2)

# param plot takeoff
plt.legend(loc='lower right')
plt.title('takeoff-aviary-v0')
plt.xlabel('Steps')
plt.ylabel('Mean Episodic Return')
plt.grid(visible=True)
plt.ticklabel_format(style='plain', axis='x')
plt.xlim(left = 0, right = 400_000)
#plt.ylim(top = 2, bottom = -1400)
'''

# Show the plot
plt.show()