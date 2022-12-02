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
num_total_steps = 1000000     # Total number of time steps to run the training
learning_rate = 1e-3            # Learning rate for optimizing the neural networks
num_epochs = 5                  # Number of epochs per time step to optimize the neural networks
epsilon = 0.4                   # Epsilon value in the PPO algorithm
trajectory_size = 1000           # Total number of trajectory samples to be sampled per time step
input_length_net = 4            # input layer size
policy_output_size = 2          # output layer size
discount_factor = 0.99
env_name = "CartPole-v1"
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
        self.dense3 = Dense(units=1, activation='linear') # 'linear' if the action space is continous

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
policy_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
value_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

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
num_passed_timesteps = 0
while num_passed_timesteps < num_total_steps:

    trajectory_observations = []
    trajectory_rewards = []
    trajectory_action_probs = []
    trajectory_advantages = []
    total_reward = 0
    total_value = 0
    #observation, info = env.reset()

    # Collect trajectory
    print("Collecting trajectory...")
    for batch_k in range(trajectory_size):
        current_action_prob = policy_net(observation.reshape(1,input_length_net))
        current_action_dist = tfd.Categorical(probs=current_action_prob)
        current_action = current_action_dist.sample().numpy()[0]

        total_value = total_value + value_net(observation.reshape((1,input_length_net)))

        # Sample new state
        observation, reward, terminated, truncated, info = env.step(current_action)

        # Collect trajectory sample
        trajectory_observations.append(observation)
        trajectory_rewards.append(reward)
        trajectory_action_probs.append(current_action_prob)
        trajectory_advantages.append(reward - value_net(observation.reshape((1,input_length_net))))
        
        if terminated or truncated:
            observation, info = env.reset(seed=42)
            break

    
    #advantage = total_reward - total_value # TODO: any other advantage estimation method can be used...
    
    # Update loop
    print("Updating the neural networks...")
    #policy_loss = 0
    #value_loss = 0
    for epoch in range(num_epochs):

        trajectory_observations = np.array(trajectory_observations)
        trajectory_action_probs = np.array(trajectory_action_probs)
        trajectory_rewards      = np.array(trajectory_rewards)
        trajectory_advantages   = np.array(trajectory_advantages)

        with tf.GradientTape() as policy_tape:
            policy_dist             = policy_net(trajectory_observations)
            #output_discrete_sampler = tfd.Categorical(probs=policy_dist)
            #policy_action           = output_discrete_sampler.sample()
            policy_action_prob      = policy_dist[:, :]
            # Policy loss update
            clip_1                  = (policy_action_prob / trajectory_action_probs) * trajectory_advantages
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
    print(f"Total rewards: {np.sum(trajectory_rewards)}")
         
env.close()
