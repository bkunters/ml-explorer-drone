import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
from mlagents_envs.environment import UnityEnvironment
import gym
tfd = tfp.distributions

#
#
#
#
#
#
#

# Parameters
unity_file_name = ""            # Unity environment name
num_total_steps = 100000    # Total number of time steps to run the training
learning_rate_policy = 1e-5            # Learning rate for optimizing the neural networks
learning_rate_value = 1e-3
num_epochs = 8                  # Number of epochs per time step to optimize the neural networks
epsilon = 0.25                   # Epsilon value in the PPO algorithm
max_trajectory_size = 10000          # max number of trajectory samples to be sampled per time step.
input_length_net = 4            # input layer size
policy_output_size = 2          # policy output layer size
discount_factor = 0.99
env_name = "CartPole-v1"        # LunarLander-v2 or MountainCar-v0 or CartPole-v1
#output_continous_sampler = tfd.MultivariateNormalDiag(loc=[0., 0., 0., 0.], scale_diag=[1., 1., 1., 1.]) # Continous output

print(f"Tensorflow version: {tf.__version__}")

#
#
#
#
#
#
#

# Define the policy network
class PolicyNetwork(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten = Flatten()
        self.dense1 = Dense(units=input_length_net, activation='tanh')
        self.dense2 = Dense(units=64, activation='tanh')
        self.dense3 = Dense(units=policy_output_size, activation='softmax') # 'linear' if the action space is continous

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

# Define the value network
class ValueNetwork(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten = Flatten()
        self.dense1 = Dense(units=input_length_net, activation='tanh')
        self.dense2 = Dense(units=64, activation='tanh')
        self.dense3 = Dense(units=1, activation='tanh')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

#
#
#
#
#
#
#

# Setup the actor/critic networks
policy_net = PolicyNetwork()
value_net = ValueNetwork()

#input = tf.constant([0,0,0,0,0,0,0,0])
#print(policy_net(input))

#
#
#
#
#
#
#

def clip_func(advantage):
    return (1+epsilon) * advantage if advantage >= 0 else (1-epsilon) * advantage

#
#
#
#
#
#
#

# This is a non-blocking call that only loads the environment.
#env = UnityEnvironment(file_name=unity_file_name, seed=42, side_channels=[])
# Start interacting with the environment.a
#env.reset()
#behavior_names = env.behavior_specs.keys()

#
#
#
#
#
#
#

# Setup training properties
policy_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_policy)
value_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_value)

env = gym.make(env_name, render_mode="human")
env.action_space.seed(42)
observation, info = env.reset(seed=42)
print(f"Observation space shape: {env.observation_space.shape}")
print(f"Action space shape: {env.action_space.shape}")
print(env.action_space)

#
#
#
#
#
#
#

# Training loop
import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = f'logs/gradient_tape/{env_name}' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

num_passed_timesteps = 0
sum_rewards = 0
num_episodes = 0
while num_passed_timesteps < num_total_steps:

    trajectory_observations = []
    trajectory_rewards = []
    trajectory_action_probs = []
    trajectory_advantages = []
    total_reward = 0
    total_value = 0
    observation, info = env.reset(seed=42)

    # Collect trajectory
    print("Collecting trajectory...")
    for batch_k in range(max_trajectory_size):
        current_action_prob = policy_net(observation.reshape(1,input_length_net))
        current_action_dist = tfd.Categorical(probs=current_action_prob)
        current_action = current_action_dist.sample().numpy()[0]

        total_value = total_value + value_net(observation.reshape((1,input_length_net)))

        # Sample new state
        observation, reward, terminated, truncated, info = env.step(current_action)
        total_reward = total_reward + reward

        # Collect trajectory sample
        trajectory_observations.append(observation)
        trajectory_rewards.append(reward)
        trajectory_action_probs.append(np.max(current_action_prob))
        
        if terminated or truncated:
            observation, info = env.reset(seed=42)
            break

    num_episodes = num_episodes + 1

    # Compute advantages
    trajectory_advantages = np.array(total_reward) - value_net(np.array(trajectory_observations))

    # Update loop
    print("Updating the neural networks...")
    for epoch in range(num_epochs):

        trajectory_observations = np.array(trajectory_observations)
        trajectory_action_probs = np.array(trajectory_action_probs)
        trajectory_rewards      = np.array(trajectory_rewards)
        trajectory_advantages   = np.array(trajectory_advantages)

        with tf.GradientTape() as policy_tape:
            policy_dist             = policy_net(trajectory_observations)
            policy_action_prob      = tf.experimental.numpy.max(policy_dist[:, :], axis=1)
            # Policy loss update
            clip_1                  = tf.multiply((tf.divide(policy_action_prob,trajectory_action_probs)),trajectory_advantages)
            clip                    = np.vectorize(clip_func)
            clip_2                  = clip(trajectory_advantages)
            policy_loss             = -tf.reduce_mean(tf.minimum(clip_1, clip_2))

        policy_gradients = policy_tape.gradient(policy_loss, policy_net.trainable_variables)
        policy_optimizer.apply_gradients(zip(policy_gradients, policy_net.trainable_variables))

        with tf.GradientTape() as value_tape:
            value_out  = tf.squeeze(value_net(trajectory_observations))
            # Value loss update
            value_loss = tf.reduce_mean(tf.square(value_out, trajectory_rewards))
            
        value_gradients = value_tape.gradient(value_loss, value_net.trainable_variables)
        value_optimizer.apply_gradients(zip(value_gradients, value_net.trainable_variables))
            
        print(f"Epoch: {epoch}, Policy loss: {policy_loss}")
        print(f"Epoch: {epoch}, Value loss: {value_loss}")


    num_passed_timesteps = num_passed_timesteps + len(trajectory_observations)

    print(f"Total time steps: {num_passed_timesteps}")
    sum_rewards = sum_rewards + np.sum(trajectory_rewards)
    mean_return = sum_rewards / num_episodes
    print(f"Mean cumulative return per episode: {mean_return}")

    # Log into tensorboard
    # TODO: @Janina could you integrate wandb here?
    with train_summary_writer.as_default():
        tf.summary.scalar('policy loss', policy_loss, step=num_episodes)
        tf.summary.scalar('value loss', value_loss, step=num_episodes)
        tf.summary.scalar('mean return', mean_return, step=num_episodes)
         
env.close()

#
#
#
#
#
#
#

# Save the policy and value networks
policy_net.save(f"{env_name}_policy_model_{num_passed_timesteps}")
value_net.save(f"{env_name}_value_model_{num_passed_timesteps}")