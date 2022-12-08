import gym

# requires gym==0.21.0 & pyglet==1.5.27
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("CartPole-v1", n_envs=1)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000, progress_bar=True)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()