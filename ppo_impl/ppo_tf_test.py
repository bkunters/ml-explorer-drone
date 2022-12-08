import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten
import tensorflow_probability as tfp
import numpy as np
from mlagents_envs.environment import UnityEnvironment
import gym
import os
from matplotlib import animation
import matplotlib.pyplot as plt

env_name = "CartPole-v1"
test_epochs = 1
input_length_net = 4

env = gym.make(env_name, render_mode="rgb_array")
observation, info = env.reset()

# Load models
policy_model = tf.keras.models.load_model(f"{env_name}_policy_model")

total_reward = 0
frames = []
for i in range(test_epochs):
    while True:
        current_action_prob = policy_model(observation.reshape(1,input_length_net))
        current_action = np.argmax(current_action_prob.numpy())

        frames.append(env.render())
        observation, reward, terminated, truncated, info = env.step(current_action)
        total_reward += reward

        if terminated or truncated:
            observation, info = env.reset()
            break
    
        

print(f"mean cumulative award per episode: {total_reward / test_epochs}")

def save_frames_as_gif(frames, path='./', filename=f'{env_name}_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

save_frames_as_gif(frames)