import gym

# requires gym==0.21.0 & pyglet==1.5.27
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
env_name = "Pendulum-v1"          # LunarLander-v2 or MountainCar-v0 or CartPole-v1 or Pendulum-v1
n_envs = 1                        # amount of envs used simultaneously

# Parallel environments
env = make_vec_env(env_name, n_envs=n_envs) # TODO: @Ardian Check stable_baseline3 library 

# Instantiate the agent
model = PPO(
    "MlpPolicy",
    env,
    gamma=0.98,
    # Using https://proceedings.mlr.press/v164/raffin22a.html
    use_sde=True,
    sde_sample_freq=4,
    learning_rate=1e-3,
    verbose=1,
)

# Train the agent
model.learn(total_timesteps=int(1e5))

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