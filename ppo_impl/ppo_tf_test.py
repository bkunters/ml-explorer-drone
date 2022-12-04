import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten
import tensorflow_probability as tfp
import numpy as np
from mlagents_envs.environment import UnityEnvironment
import gym

env_name = "CartPole-v1"
test_epochs = 100
input_length_net = 4

env = gym.make(env_name, render_mode="human")
observation, info = env.reset()

# Load models
policy_model = tf.keras.models.load_model(f"{env_name}_policy_model")

total_reward = 0
while True:
    current_action_prob = policy_model(observation.reshape(1,input_length_net))
    current_action = np.argmax(current_action_prob.numpy())

    observation, reward, terminated, truncated, info = env.step(current_action)
    total_reward += reward

    if terminated or truncated:
        observation, info = env.reset()
        total_reward = 0
