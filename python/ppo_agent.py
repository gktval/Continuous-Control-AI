import numpy as np

from utils.ppo_replay import PPOBuffer
import model
from utils.noise import OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from collections import deque

class Agent:
    def __init__(self, state_size, action_size, config):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = config.seed
        self.buffer_size = config.EXP_REPLAY_SIZE
        self.gamma = config.GAMMA
        self.batch_size = config.BATCH_SIZE
        self.update_every = config.UPDATE_FREQ
        self.lr_actor = config.LR_Actor
        self.device = config.device
        self.weight_decay = config.WEIGHT_DECAY
        self.eps_clip = config.EPS_CLIP
        self.optimization_steps = config.OPT_STEPS
        self.opt_eps = config.OPT_EPS
        self.gae_lambda=config.GAE_LAMBDA
        self.gradient_clip = config.GRAD_CLIP
        self.entropy_penalty_weight = config.ENT_PENALTY
        self.loss_weight = config.LOSS_WEIGHT

        self.std_scale = 1.0

         # Network
        self.network = model.GaussianActorCriticNet(
            state_size=state_size,
            action_size=action_size,
            shared_layers=[512,512],
            actor_layers=[512],
            critic_layers=[512],
            std_init=0
        )
        self.network.to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.lr_actor,
            eps=self.opt_eps,
            weight_decay=self.weight_decay
        )
       
        # Replay memory
        self.memory = PPOBuffer(self.buffer_size, self.batch_size, self.device, self.seed)
       
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0


    # Set learning rates
    def set_learning_rate(self, lr_actor):
        self.learning_rate_actor = lr_actor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_actor


    def act(self, states):
        # state should be transformed to a tensor
        numpyState = torch.from_numpy(np.array(states))
        state = numpyState.float().to(self.device) 

        values, action, log_probs, entropy = self.network(state, std_scale=self.std_scale)

        return action.cpu().numpy(), log_probs.cpu().detach().numpy(), values.cpu().detach().numpy()


    def step(self, states, actions, rewards, next_states, dones, log_probs, values):
        self.memory.add(states, actions, rewards, next_states, dones, log_probs, values)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and self.memory.is_ready():
            trajectories = self.collect_trajectories(next_states)
            self.learn(trajectories)
            self.memory.clear()


    def collect_trajectories(self, next_states):
        """Collect a set of trajectories for each agent.
        :param next_states:  Final state to start rollback from
        :return:  (dict) Dictionary containing tensors for states, actions,
            log_probs, returns, and advantages
        """        
        stacked_next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)
        value, action, log_prob, entropy = self.network(stacked_next_states)
        returns = value.cpu().detach().numpy()

        p_next_values = returns
        advantages = np.zeros(len(next_states))

        _states = []
        _actions = []
        _logProbs = []
        _returns = []
        _advantages = []

        # Perform rollback on trajectories
        for experience in self.memory.reverse_iterator():
            states, actions, rewards, nextStates, endFlags, logProbs, values = experience
            endFlags = np.vstack([endFlags]).astype(np.uint8)[0]
            rewards = np.array(rewards)

            returns = rewards * self.gamma * (1-endFlags) * returns
            td_error = rewards + self.gamma * (1-endFlags) * p_next_values - values
            advantages = advantages * self.gae_lambda * self.gamma * (1-endFlags) + td_error

            _states.insert(0,states)
            _actions.insert(0,actions)
            _logProbs.insert(0,logProbs)
            _returns.insert(0,returns)
            _advantages.insert(0,advantages)
            p_next_values = values

        _states = np.stack(_states)
        _actions= np.stack(_actions)
        _logProbs= np.stack(_logProbs)
        _returns= np.stack(_returns)
        _advantages= np.stack(_advantages)
        
        # Normalize advantages
        _advantages = (_advantages - _advantages.mean())/_advantages.std()

        return (_states,_actions,_logProbs,_returns,_advantages)
    

    def learn(self, trajectories):
        """Perform optimization steps on network over samples from the provided trajectory.
        :param trajectories:  (dict) Dictionary containing tensors for states, actions,
            log_probs, returns, and advantages
        """
        states, actions, log_probs, returns, advantages = list(trajectories)

        trajectory_size = states.shape[0]

        # For each optimization step, collect a random batch of indexes and propagate loss
        for _ in range(self.optimization_steps):
            
            batch_indices = torch.randint(low=0, high=trajectory_size, size=(self.batch_size,)).long()

            sampled_states = torch.from_numpy(states[batch_indices]).float().to(self.device)
            sampled_actions = torch.from_numpy(actions[batch_indices]).float().to(self.device)
            sampled_log_probs = torch.from_numpy(log_probs[batch_indices]).float().to(self.device)
            sampled_returns = torch.from_numpy(returns[batch_indices]).float().to(self.device)
            sampled_advantages = torch.from_numpy(advantages[batch_indices]).float().to(self.device)

            L = self.loss(sampled_states, sampled_actions, sampled_log_probs, sampled_returns, sampled_advantages)

            self.optimizer.zero_grad()
            clip_grad_norm_(self.network.parameters(), self.gradient_clip)
            L.backward()
            self.optimizer.step()

            del L

    def loss(self, states, actions, log_probs, rewards, advantages):
        """Compute the loss function for a sampled batch of experiences
        :param states:  (tensor) Sampled states
        :param actions:  (tensor) Sampled actions
        :param log_probs:  (tensor) Sampled log probabilities
        :param rewards:  (tensor) Sampled propagated rewards
        :param advantages:  (tensor) Sampled propagated advantages
        :return:  Loss function for optimizer
        """

        # Compute predictions from states, actions to get value for critic loss and log probabilities for entropy
        value, _, log_probs_old, entropy = self.network(states, actions)

        # Ratio is product of probability ratios -- store as log probabilities instead
        ratio = (log_probs_old - log_probs).sum(-1).exp()

        # Compute clipped policy loss (L_clip)
        loss_original = ratio * advantages
        loss_clipped = ratio.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
        policy_loss = -torch.min(loss_original, loss_clipped).mean()

        # Apply penalty for entropy
        entropy_penalty = -self.entropy_penalty_weight * entropy.mean()

        # Compute value function loss (mean squared error)
        value_loss = self.loss_weight * F.mse_loss(value.view(-1), rewards.view(-1))

        return policy_loss + entropy_penalty + value_loss