import datetime
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten
import tensorflow_probability as tfp
import numpy as np
# from mlagents_envs.environment import UnityEnvironment
import gym
import wandb

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
num_total_steps = 25e3           # Total number of time steps to run the training
learning_rate_policy = 1e-3     # Learning rate for optimizing the neural networks
learning_rate_value = 1e-3
num_epochs = 5                  # Number of epochs per time step to optimize the neural networks
epsilon = 0.2                   # Epsilon value in the PPO algorithm
max_trajectory_size = 10000     # max number of trajectory samples to be sampled per time step. 
trajectory_iterations = 16      # number of batches of episodes
input_length_net = 4            # input layer size
policy_output_size = 2          # policy output layer size
discount_factor = 0.99
env_name = "CartPole-v1"        # LunarLander-v2 or MountainCar-v0 or CartPole-v1

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
        self.dense3 = Dense(units=64, activation='tanh')
        self.out = Dense(units=policy_output_size, activation='softmax') # 'linear' if the action space is continous

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.out(x)

# Define the value network
class ValueNetwork(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten = Flatten()
        self.dense1 = Dense(units=input_length_net, activation='tanh')
        self.dense2 = Dense(units=64, activation='tanh')
        self.dense3 = Dense(units=64, activation='tanh')
        self.out = Dense(units=1, activation='tanh')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.out(x)

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

env = gym.make(env_name)
env.action_space.seed(42)
env.observation_space.seed(42)
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
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# logging
train_log_dir = f'logs/gradient_tape/{env_name}' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
wandb.init(
    project=f'drone-mechanics-ppo',
    entity='drone-mechanics',
    sync_tensorboard=True,
    config={ # stores hyperparams in job
            'total epochs': num_epochs,
            'total steps': num_total_steps,
            'batches per episode': trajectory_iterations,
            'input layer size': input_length_net,
            'output layer size': policy_output_size,
            'lr policyNet': learning_rate_policy,
            'lr valueNet': learning_rate_value,
            'epsilon': epsilon,
            'discount': discount_factor
    },
    name=f"{env_name}__{current_time}",
    # monitor_gym=True,
    save_code=True,
)

num_passed_timesteps = 0
sum_rewards = 0
num_episodes = 0
last_mean_reward = 0
while num_passed_timesteps < num_total_steps:

    trajectory_observations = []
    trajectory_rewards = []
    trajectory_action_probs = []
    trajectory_advantages = []
    total_reward = 0
    total_value = 0
    observation, info = env.reset(seed=42)

    # Collect trajectory
    sum_rewards = 0
    print("Collecting batch trajectories...")
    for _ in range(trajectory_iterations):

        # collect samples until agent craches
        for batch_k in range(max_trajectory_size):
            current_action_prob = policy_net(observation.reshape(1,input_length_net))
            current_action_dist = tfd.Categorical(probs=current_action_prob)
            current_action = current_action_dist.sample(seed=42).numpy()[0]

            total_value = total_value + value_net(observation.reshape((1,input_length_net)))

            # Sample new state etc. from environment
            observation, reward, terminated, truncated, info = env.step(current_action)
            total_reward = total_reward + reward

            # Collect trajectory sample
            trajectory_observations.append(observation)
            trajectory_rewards.append(reward)
            trajectory_action_probs.append(np.max(current_action_prob))
        
            if terminated or truncated:
                observation, info = env.reset()
                break

    num_episodes = num_episodes + 1

    # Compute advantages
    trajectory_advantages = np.array(total_reward) - value_net(np.array(trajectory_observations))

    # Update the network loop
    print("Updating the neural networks...")
    for epoch in range(num_epochs):

        trajectory_observations = np.array(trajectory_observations)
        trajectory_action_probs = np.array(trajectory_action_probs)
        trajectory_rewards      = np.array(trajectory_rewards)
        trajectory_advantages   = np.array(trajectory_advantages)

        # Compute policy loss
        with tf.GradientTape() as policy_tape:
            policy_dist             = policy_net(trajectory_observations)
            policy_action_prob      = tf.experimental.numpy.max(policy_dist[:, :], axis=1)
            # Policy loss update
            ratios                  = tf.divide(policy_action_prob,tf.constant(trajectory_action_probs))
            clip_1                  = tf.multiply(ratios,tf.squeeze(trajectory_advantages))
            clip                    = tf.clip_by_value(ratios, 1.0-epsilon, 1.0+epsilon)
            clip_2                  = tf.multiply(clip, tf.squeeze(trajectory_advantages))
            min_val                 = tf.minimum(clip_1, clip_2)
            policy_loss             = tf.math.negative(tf.reduce_mean(min_val))

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
    mean_return = sum_rewards / (trajectory_iterations)
    print(f"Mean cumulative return per episode: {mean_return}")

    # Make sure the best model is saved.
    if mean_return > last_mean_reward:
        # Save the policy and value networks for further training/tests
        policy_net.save(f"{env_name}_policy_model")
        value_net.save(f"{env_name}_value_model")
        last_mean_reward = mean_return

    # Log into tensorboard & Wandb
    wandb.log({
        'step': num_episodes,
        'timesteps': num_passed_timesteps, # die Summe der Episoden über die Gesamte Episoden Anzahl
        'policy loss': policy_loss, 
        'value loss': value_loss, 
        'mean reward': mean_return, 
        'sum rewards': sum_rewards})

    with train_summary_writer.as_default():
        tf.summary.scalar('policy loss', policy_loss, step=num_episodes)
        tf.summary.scalar('value loss', value_loss, step=num_episodes)
        tf.summary.scalar('mean return', mean_return, step=num_episodes)
   
env.close()
wandb.run.finish() if wandb and wandb.run else None
#
#
#
#
#
#
#

# Save the policy and value networks for further training/tests
policy_net.save(f"{env_name}_policy_model_{num_passed_timesteps}")
value_net.save(f"{env_name}_value_model_{num_passed_timesteps}")