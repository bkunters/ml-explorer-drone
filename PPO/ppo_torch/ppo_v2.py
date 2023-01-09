from collections import deque
import datetime
import torch
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import numpy as np

# logging python
import logging
import sys

# monitoring/logging ML
import wandb

from collections import deque
from PPO.ppo_torch.ppo_continuous import PolicyNet, ValueNet

# Paths and other constants
MODEL_PATH = './models/'
LOG_PATH = './log/'
VIDEO_PATH = './video/'
RESULTS_PATH = './results/'

CURR_DATE = datetime.today().strftime('%Y-%m-%d')


class PPO_PolicyGradient_V2:
    """ Proximal Policy Optimization (PPO) is an online policy gradient method.
        As an online policy method it updates the policy and then discards the experience (no replay buffer).
        Thus the agent does well in environments with dense reward signals.
        The clipped objective function in PPO allows to keep the policy close to the policy 
        that was used to sample the data resulting in a more stable training. 
    """
    # Further reading
    # PPO experiments: https://nn.labml.ai/rl/ppo/experiment.html
    #                  https://nn.labml.ai/rl/ppo/index.html
    # PPO explained:   https://huggingface.co/blog/deep-rl-ppo

    def __init__(self, 
        env, 
        in_dim, 
        out_dim,
        total_steps,
        max_trajectory_size,
        trajectory_iterations,
        noptepochs=5,
        lr_p=1e-3,
        lr_v=1e-3,
        gae_lambda=0.95,
        gamma=0.99,
        epsilon=0.22,
        adam_eps=1e-5,
        render=10,
        save_model=10,
        csv_writer=None,
        stats_plotter=None,
        log_video=False) -> None:
        
        # hyperparams
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.total_steps = total_steps
        self.max_trajectory_size = max_trajectory_size
        self.trajectory_iterations = trajectory_iterations
        self.noptepochs = noptepochs
        self.lr_p = lr_p
        self.lr_v = lr_v
        self.gamma = gamma
        self.epsilon = epsilon
        self.adam_eps = adam_eps
        self.gae_lambda = gae_lambda

        # environment
        self.env = env
        self.render_steps = render
        self.save_model = save_model

        # track video of gym
        self.log_video = log_video

        # keep track of rewards per episode
        self.ep_returns = deque(maxlen=max_trajectory_size)
        self.csv_writer = csv_writer
        self.stats_plotter = stats_plotter
        self.stats_data = {'mean episodic length': [], 'mean episodic rewards': [], 'timestep': []}

        # add net for actor and critic
        self.policy_net = PolicyNet(self.in_dim, self.out_dim) # Setup Policy Network (Actor) - (policy-based method) "How the agent behaves"
        self.value_net = ValueNet(self.in_dim, 1) # Setup Value Network (Critic) -  (value-based method) "How good the action taken is."

        # add optimizer for actor and critic
        self.policy_net_optim = Adam(self.policy_net.parameters(), lr=self.lr_p, eps=self.adam_eps) # Setup Policy Network (Actor) optimizer
        self.value_net_optim = Adam(self.value_net.parameters(), lr=self.lr_v, eps=self.adam_eps)  # Setup Value Network (Critic) optimizer

    def get_continuous_policy(self, obs):
        """Make function to compute action distribution in continuous action space."""
        # Multivariate Normal Distribution Lecture 15.7 (Andrew Ng) https://www.youtube.com/watch?v=JjB58InuTqM
        # fixes the detection of outliers, allows to capture correlation between features
        # https://discuss.pytorch.org/t/understanding-log-prob-for-normal-distribution-in-pytorch/73809
        # 1) Use Normal distribution for continuous space
        action_prob = self.policy_net(obs) # query Policy Network (Actor) for mean action
        cov_matrix = torch.diag(torch.full(size=(self.out_dim,), fill_value=0.5))
        return MultivariateNormal(action_prob, covariance_matrix=cov_matrix)

    def get_action(self, dist):
        """Make action selection function (outputs actions, sampled from policy)."""
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy
    
    def get_values(self, obs, actions):
        """Make value selection function (outputs values for obs in a batch)."""
        values = self.value_net(obs).squeeze()
        dist = self.get_continuous_policy(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_prob, entropy

    def get_value(self, obs):
        return self.value_net(obs).squeeze()

    def step(self, obs):
        """ Given an observation, get action and probabilities from policy network (actor)"""
        action_dist = self.get_continuous_policy(obs) 
        action, log_prob, entropy = self.get_action(action_dist)
        return action.detach().numpy(), log_prob.detach().numpy(), entropy.detach().numpy()

    def generalized_advantage_estimate_0(self, batch_rewards, values, normalized=True):
        """ Generalized Advantage Estimate calculation
            Calculate delta, which is defined by delta = r - v 
        """
        # Cumulative rewards: https://gongybable.medium.com/reinforcement-learning-introduction-609040c8be36
        # return value: G(t) = R(t) + gamma * R(t-1)
        cum_returns = []
        for rewards in reversed(batch_rewards): # reversed order
            discounted_reward = 0
            for reward in reversed(rewards):
                discounted_reward = reward + (self.gamma * discounted_reward)
                cum_returns.insert(0, discounted_reward) # reverse it again
        advantages = cum_returns - values # delta = r - v
        if normalized:
            advantages = self.normalize_adv(advantages)
        return advantages

    def generalized_advantage_estimate_1(self, batch_rewards, values, normalized=True):
        """ Generalized Advantage Estimate calculation
            - GAE defines advantage as a weighted average of A_t
            - advantage measures if an action is better or worse than the policy's default behavior
            - want to find the maximum Advantage representing the benefit of choosing a specific action
        """
        # check if tensor and convert to numpy
        if torch.is_tensor(batch_rewards):
            batch_rewards = batch_rewards.detach().numpy()
        if torch.is_tensor(values):
            values = values.detach().numpy()

        # STEP 4: compute returns as G(t) = R(t) + gamma * R(t-1)
        # STEP 5: compute advantage estimates δ_t = − V(s_t) + r_t
        cum_returns = []
        advantages = []
        for rewards in reversed(batch_rewards): # reversed order
            discounted_reward = 0
            for i in reversed(range(len(rewards))):
                discounted_reward = rewards[i] + (self.gamma * discounted_reward)
                # Hinweis @Thomy Delta Könnte als advantage ausreichen
                # δ_t = − V(s_t) + r_t
                delta = discounted_reward - values[i] # delta = r - v
                advantages.insert(0, delta)
                cum_returns.insert(0, discounted_reward) # reverse it again

        # convert numpy to torch tensor
        cum_returns = torch.tensor(np.array(cum_returns), dtype=torch.float)
        advantages = torch.tensor(np.array(advantages), dtype=torch.float)
        if normalized:
            advantages = self.normalize_adv(advantages)
        return advantages, cum_returns


    def generalized_advantage_estimate_2(self, obs, next_obs, batch_rewards, dones, normalized=True):
        """ Generalized Advantage Estimate calculation
            - GAE defines advantage as a weighted average of A_t
            - advantage measures if an action is better or worse than the policy's default behavior
            - want to find the maximum Advantage representing the benefit of choosing a specific action
        """
        # general advantage estimage paper: https://arxiv.org/pdf/1506.02438.pdf
        # general advantage estimage other: https://nn.labml.ai/rl/ppo/gae.html

        s_values = self.get_value(obs).detach().numpy()
        ns_values = self.get_value(next_obs).detach().numpy()
        advantages = []
        returns = []

        # STEP 4: Calculate cummulated reward
        for rewards in reversed(batch_rewards):
            prev_advantage = 0
            returns_current = ns_values[-1]  # V(s_t+1)
            for i in reversed(range(len(rewards))):
                # STEP 5: compute advantage estimates A_t at step t
                mask = (1.0 - dones[i])
                gamma = self.gamma * mask
                td_error = rewards[i] + gamma * ns_values[i] - s_values[i]
                # A_t = δ_t + γ * λ * A(t+1)
                prev_advantage = td_error + gamma * self.gae_lambda * prev_advantage
                returns_current = rewards[i] + gamma * returns_current
                # reverse it again
                returns.insert(0, returns_current)
                advantages.insert(0, prev_advantage)
        advantages = np.array(advantages)
        if normalized:
            advantages = self.normalize_adv(advantages)
        return torch.tensor(np.array(advantages), dtype=torch.float), torch.tensor(np.array(returns), dtype=torch.float)


    def generalized_advantage_estimate_3(self, batch_rewards, values, dones, normalized=True):
        """ Calculate advantage as a weighted average of A_t
                - advantage measures if an action is better or worse than the policy's default behavior
                - GAE allows to balance bias and variance through a weighted average of A_t

                - gamma (dicount factor): allows reduce variance by downweighting rewards that correspond to delayed effects
                - done (Tensor): boolean flag for end of episode. TODO: Q&A
        """
        # general advantage estimage paper: https://arxiv.org/pdf/1506.02438.pdf
        # general advantage estimage other: https://nn.labml.ai/rl/ppo/gae.html

        advantages = []
        returns = []
        values = values.detach().numpy()
        for rewards in reversed(batch_rewards): # reversed order
            prev_advantage = 0
            discounted_reward = 0
            last_value = values[-1] # V(s_t+1)
            for i in reversed(range(len(rewards))):
                # TODO: Q&A handling of special cases GAE(γ, 0) and GAE(γ, 1)
                # bei Vetorisierung, bei kurzen Episoden (done flag)
                # mask if episode completed after step i 
                mask = 1.0 - dones[i] 
                last_value = last_value * mask
                prev_advantage = prev_advantage * mask

                # TD residual of V with discount gamma
                # δ_t = − V(s_t) + r_t + γ * V(s_t+1)
                # TODO: Delta Könnte als advantage ausreichen, r - v könnte ausreichen
                delta = - values[i] + rewards[i] + (self.gamma * last_value)
                # discounted sum of Bellman residual term
                # A_t = δ_t + γ * λ * A(t+1)
                prev_advantage = delta + self.gamma * self.gae_lambda * prev_advantage
                discounted_reward = rewards[i] + (self.gamma * discounted_reward)
                returns.insert(0, discounted_reward) # reverse it again
                advantages.insert(0, prev_advantage) # reverse it again
                # store current value as V(s_t+1)
                last_value = values[i]
        advantages = torch.tensor(np.array(advantages), dtype=torch.float)
        returns = torch.tensor(np.array(returns), dtype=torch.float)
        if normalized:
            advantages = self.normalize_adv(advantages)
        return advantages, returns


    def normalize_adv(self, advantages):
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    def normalize_ret(self, returns):
        return (returns - returns.mean()) / returns.std()

    def finish_episode(self):
        pass 

    def collect_rollout(self, n_step=1, render=True):
        """Collect a batch of simulated data each time we iterate the actor/critic network (on-policy)"""
        
        step, trajectory_rewards = 0, []

        # collect trajectories
        trajectory_obs = []
        trajectory_nextobs = []
        trajectory_actions = []
        trajectory_action_probs = []
        trajectory_dones = []
        batch_rewards = []
        batch_lens = []

        # Run Monte Carlo simulation for n timesteps per batch
        logging.info("Collecting batch trajectories...")
        while step < n_step:
            
            # rewards collected per episode
            trajectory_rewards, done = [], False 
            obs = self.env.reset()

            # Run episode for a fixed amount of timesteps
            # to keep rollout size fixed and episodes independent
            for ep_t in range(0, self.max_trajectory_size):
                # render gym envs
                if render and ep_t % self.render_steps == 0:
                    self.env.render()
                
                step += 1 

                # action logic 
                # sampled via policy which defines behavioral strategy of an agent
                action, log_probability, _ = self.step(obs)
                        
                # STEP 3: collecting set of trajectories D_k by running action 
                # that was sampled from policy in environment
                __obs, reward, done, info = self.env.step(action)

                # collection of trajectories in batches
                trajectory_obs.append(obs)
                trajectory_nextobs.append(__obs)
                trajectory_actions.append(action)
                trajectory_action_probs.append(log_probability)
                trajectory_rewards.append(reward)
                trajectory_dones.append(done)
                    
                obs = __obs

                # break out of loop if episode is terminated
                if done:
                    break
            
            batch_lens.append(ep_t + 1) # as we started at 0
            batch_rewards.append(trajectory_rewards)

        # convert trajectories to torch tensors
        obs = torch.tensor(np.array(trajectory_obs), dtype=torch.float)
        next_obs = torch.tensor(np.array(trajectory_nextobs), dtype=torch.float)
        actions = torch.tensor(np.array(trajectory_actions), dtype=torch.float)
        action_log_probs = torch.tensor(np.array(trajectory_action_probs), dtype=torch.float)
        dones = torch.tensor(np.array(trajectory_dones), dtype=torch.float)

        return obs, next_obs, actions, action_log_probs, dones, batch_rewards, batch_lens
                

    def train(self, values, returns, advantages, batch_log_probs, curr_log_probs, epsilon):
        """Calculate loss and update weights of both networks."""
        logging.info("Updating network parameter...")
        # loss of the policy network
        self.policy_net_optim.zero_grad() # reset optimizer
        policy_loss = self.policy_net.loss(advantages, batch_log_probs, curr_log_probs, epsilon)
        policy_loss.backward() # backpropagation
        self.policy_net_optim.step() # single optimization step (updates parameter)

        # loss of the value network
        self.value_net_optim.zero_grad() # reset optimizer
        value_loss = self.value_net.loss(values, returns)
        value_loss.backward()
        self.value_net_optim.step()

        return policy_loss, value_loss

    def learn(self):
        """"""
        steps = 0

        while steps < self.total_steps:
            policy_losses, value_losses = [], []
            # Collect trajectory
            # STEP 3: simulate and collect trajectories --> the following values are all per batch
            obs, next_obs, actions, batch_log_probs, dones, rewards, batch_lens = self.collect_rollout(n_step=self.trajectory_iterations)

            # timesteps simulated so far for batch collection
            steps += np.sum(batch_lens)

            # STEP 4-5: Calculate cummulated reward and GAE at timestep t_step
            values, _ , _ = self.get_values(obs, actions)
            # cum_returns = self.cummulative_return(rewards)
            # advantages = self.advantage_estimate_(cum_returns, values.detach())
            advantages, cum_returns = self.generalized_advantage_estimate_1(rewards, values.detach())

            # update network params 
            for _ in range(self.noptepochs):
                # STEP 6-7: calculate loss and update weights
                values, curr_log_probs, _ = self.get_values(obs, actions)
                policy_loss, value_loss = self.train(values, cum_returns, advantages, batch_log_probs, curr_log_probs, self.epsilon)
                
                policy_losses.append(policy_loss.detach().numpy())
                value_losses.append(value_loss.detach().numpy())

            # log all statistical values to CSV
            self.log_stats(policy_losses, value_losses, rewards, batch_lens, steps)

            # store model in checkpoints
            if steps % self.save_model == 0:
                env_name = env.unwrapped.spec.id
                policy_net_name = f'{MODEL_PATH}{env_name}_{CURR_DATE}_policyNet.pth'
                value_net_name = f'{MODEL_PATH}{env_name}_{CURR_DATE}_valueNet.pth'
                torch.save({
                    'epoch': steps,
                    'model_state_dict': self.policy_net.state_dict(),
                    'optimizer_state_dict': self.policy_net_optim.state_dict(),
                    'loss': policy_loss,
                    }, policy_net_name)
                torch.save({
                    'epoch': steps,
                    'model_state_dict': self.value_net.state_dict(),
                    'optimizer_state_dict': self.value_net_optim.state_dict(),
                    'loss': value_loss,
                    }, value_net_name)

                if wandb:
                    wandb.save(policy_net_name)
                    wandb.save(value_net_name)

                # Log to CSV
                if self.csv_writer:
                    self.csv_writer(self.stats_data)
                    for value in self.stats_data.values():
                        del value[:]

        # Finalize and plot stats
        if self.stats_plotter:
            df = self.stats_plotter.read_csv()
            self.stats_plotter.plot(df, x='timestep', y='mean episodic rewards', title=env_name)

    def log_stats(self, p_losses, v_losses, batch_return, batch_lens, steps):
        """Calculate stats and log to W&B, CSV, logger """
        if torch.is_tensor(batch_return):
            batch_return = batch_return.detach().numpy()
        # calculate stats
        mean_p_loss = np.mean([np.sum(loss) for loss in p_losses])
        mean_v_loss = np.mean([np.sum(loss) for loss in v_losses])

        # Calculate the stats of an episode
        cum_ret = [np.sum(ep_rews) for ep_rews in batch_return]
        mean_ep_lens = np.mean(batch_lens)
        mean_ep_rews = np.mean(cum_ret)
        # calculate standard deviation (spred of distribution)
        std_ep_rews = np.std(cum_ret)

        # Log stats to CSV file
        self.stats_data['mean episodic length'].append(mean_ep_lens)
        self.stats_data['mean episodic rewards'].append(mean_ep_rews)
        self.stats_data['timestep'].append(steps)

        # Monitoring via W&B
        wandb.log({
            'train/timesteps': steps,
            'train/mean policy loss': mean_p_loss,
            'train/mean value loss': mean_v_loss,
            'train/mean episode length': mean_ep_lens,
            'train/mean episode returns': mean_ep_rews,
            'train/std episode returns': std_ep_rews
        })

        logging.info('\n')
        logging.info(f'------------ Episode: {steps} --------------')
        logging.info(f"Mean return:          {mean_ep_rews}")
        logging.info(f"Mean policy loss:     {mean_p_loss}")
        logging.info(f"Mean value loss:      {mean_v_loss}")
        logging.info('--------------------------------------------')
        logging.info('\n')

