# ML Explorer Drone - Quickstart

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

For video install:
- Ubuntu: ```$ sudo apt install ffmpeg```
- Windows: https://github.com/utiasDSL/gym-pybullet-drones/tree/master/assignments#on-windows

Then register the custom environemnts:
- ```$ cd gym-pybullet-drones/```
- ```$ pip install -e .```

[2] Test if everything runs
- ```$ cd gym_pybullet_drones/gym_pybullet_drones/examples```
- run ```>>> python fly.py```

## 3. custom implementation [PPOV1, PPV2]

[1] Installation

- ```$ conda create --name ppo_py310 python=3.10```
- ```$ conda activate ppo_py310```
- ```$ pip install -r requirements.txt```

[2] Training

Run training with PPO
- ```>>> python ppo_continuous.py --train True```

[3] Evaluation
- ```>>> python ppo_continuous.py --test True```

[3] Hyperparameter Tuning
Run training with PPO
- ```>>> python ppo_continuous.py --hyperparam True```