# ML Explorer Drone - Quickstart

This repository contains a UnityML drone environment and the PyBullet Drone environment as well as a custom PPO algorithm.

### Top level directory layout
.
├── gym_pybullet_drones                 # PyBullet drone environment
├── ml_explorer_drones                  # UnityML explorer environment (for further development)
├── src                                 # Source files 
├── results                             # Results including models, logs, plots
├── tools                               # Tools and utilities
├── requirements.txt                    # requirements.txt for pybullet and ppo_v2
├── LICENSE
└── README.md

## 1. Unity [ml-explorer-drone environment]
Unity Machine Learning Controlled Explorer Drone
Will be continued in the future...

## 2. PyBullet Drones [gym-pybullet-drone environment]

[1] Installation

Create conda env
- ```$ conda create --name ppo_drone_py39 python=3.9```
- ```$ conda activate ppo_drone_py39```

Installation
- ```$ pip install -r requirements_pybullet.txt```
or 
- ```$ pip install --upgrade numpy Pillow matplotlib cycler```
- ```$ pip install --upgrade gym pybullet stable_baselines3 'ray[rllib]'```

For video and graphical output install:
- Ubuntu: ```$ sudo apt install ffmpeg```
- Windows: https://github.com/utiasDSL/gym-pybullet-drones/tree/master/assignments#on-windows

Then register the custom PyBullet environemnts:
- ```$ cd gym-pybullet-drones/```
- ```$ pip install -e .```

[2] Test if everything runs
- ```$ cd gym_pybullet_drones/examples```
- ```$ python fly.py```

[3] Training
The default is the custom _PPO_v2_ algorithm with takeoff-aviary-v0 environment.
Select a training from scratch with the flags:

- ```$ python learn.py --env_id "takeoff" --algo "ppo_v2"```
- ```$ python learn.py --env_id "hover" --algo "ppo_v2"```


## 3. Custom implementation [PPO_V1, PPO_V2]

[1] Installation

- ```$ conda create --name ppo_py310 python=3.10```
- ```$ conda activate ppo_py310```
- ```$ pip install -r requirements.txt```

[2] Training

Run training with PPO
- ```python ppo_continuous.py --train True```

[3] Evaluation
- ```python ppo_continuous.py --test True```

[3] Hyperparameter Tuning
Run training with PPO
- ```python ppo_continuous.py --hyperparam True```