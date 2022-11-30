import tensorflow as tf
from tensorflow.python.keras.layers import Dense
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
from mlagents_envs.environment import UnityEnvironment
import gym
tfd = tfp.distributions

# Parameters
unity_file_name = ""
num_steps = 100000000
learning_rate = 1e-3
num_epochs = 10
epsilon = 0.2
trajectory_size = 1000000
input_length_net = 8
#output_sampler = tfd.MultivariateNormalDiag(loc=[0., 0., 0., 0.], scale_diag=[1., 1., 1., 1.]) # Continous output

print(f"Tensorflow version: {tf.__version__}")

# Define the neural network
class Network(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = Dense(units=input_length_net, activation='relu')
        self.dense2 = Dense(units=64, activation='relu')
        self.dense3 = Dense(units=4, activation='softmax') # 'linear' if the action space is continous

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


# Setup the actor/critic networks
policy_net = Network()
value_net = Network()

def ppo_clip():
    pass

def collect_trajectories():
    pass

def compute_rewards():
    pass

def compute_advantage():
    pass

def update_policy():
    pass

def update_value():
    pass

def policy_loss_fn(y_true, y_pred):
    loss = tf.minimum(y_true, y_pred)
    return -tf.reduce_mean(loss)

def value_loss_fn(y_true, y_pred):
    loss = tf.square(y_true-y_pred)
    return tf.reduce_mean(loss)

def clip_func(advantage):
    return (1+epsilon) * advantage if advantage >= 0 else (1-epsilon) * advantage



# This is a non-blocking call that only loads the environment.
#env = UnityEnvironment(file_name=unity_file_name, seed=42, side_channels=[])
# Start interacting with the environment.a
#env.reset()
#behavior_names = env.behavior_specs.keys()

# Setup training properties
policy_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
value_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

env = gym.make("LunarLander-v2", render_mode="human")
env.action_space.seed(42)
observation, info = env.reset(seed=42)
print(f"Observation space shape: {env.observation_space.shape}")
print(f"Action space shape: {env.action_space.shape}")

print(env.action_space)
for i in range(num_steps):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    #print(observation)
    #print(reward)
    #print(env.action_space.sample())

    value_out = value_net(observation)
    advantage = total_reward - value_out

    '''
    # TODO: test the ppo algorithm
    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            policy_out = policy_net(observation)
            value_out = value_net(observation)

            # Policy update
            clip_1 = (policy_out / policy_i) * advantage # TODO: policy_i was computed while the trajectory was computed.
            clip_2 = clip_func(advantage)
            policy_loss = policy_loss_fn(clip_1, clip_2)

            # Value update
            value_loss = value_loss_fn(value_out, total_reward)

        policy_gradients = tape.gradient(policy_loss, policy_net.trainable_variables)
        policy_optimizer.apply_gradients(zip(policy_gradients, policy_net.trainable_variables))
        value_gradients = tape.gradient(value_loss, value_net.trainable_variables)
        value_optimizer.apply_gradients(zip(value_gradients, value_net.trainable_variables))
    '''
        

    if terminated or truncated:
        observation, info = env.reset()
        
        
env.close()
