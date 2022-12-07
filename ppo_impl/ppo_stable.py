import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


# TODO: Hyperparameters should be the same in baseline and our own implementation

# # Parameters
# num_total_steps = 25e3           # Total number of time steps to run the training
# learning_rate_policy = 1e-3     # Learning rate for optimizing the neural networks
# learning_rate_value = 1e-3
# num_epochs = 5                  # Number of epochs per time step to optimize the neural networks
# epsilon = 0.2                   # Epsilon value in the PPO algorithm
# max_trajectory_size = 10000     # max number of trajectory samples to be sampled per time step. 
# trajectory_iterations = 16      # number of batches of episodes
# input_length_net = 4            # input layer size
# policy_output_size = 2          # policy output layer size
# discount_factor = 0.99
# env_name = "CartPole-v1-StableBaseline"  # LunarLander-v2 or MountainCar-v0 or CartPole-v1
# n_envs = 1                               # amount of envs used simultaneously

# Parallel environments
env_name = "CartPole-v1" # 'Pendulum'
env = make_vec_env(env_name, n_envs=1) # TODO: @Ardian Check stable_baseline3 library 

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

    # TODO: Define values to track --> look into PPO
    # TODO: Track the value in W&B (ppo_tf.py is already implemented) @Ardian
    # TODO: What is the maximum reward we can reach --> 500 

    # Log into tensorboard & Wandb
    # wandb.log({
    #     'step': num_episodes,
    #     'timesteps': num_passed_timesteps,
    #     'policy loss': policy_loss, 
    #     'value loss': value_loss, 
    #     'mean reward': mean_return, 
    #     'sum rewards': sum_rewards})