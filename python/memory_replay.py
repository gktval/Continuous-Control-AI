import random
from collections import namedtuple, deque
import numpy as np
import torch

class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, device):
        self.__buffer_size = buffer_size
        self.__batch_size = batch_size
        self.__device = device

        self.__experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.__memory = deque(maxlen=buffer_size)

    def get_batch_size(self):
        return self.__batch_size

    def is_ready(self):
        return len(self) >= self.__batch_size

    def add(self, state, action, reward, next_state, done):
        self.__memory.append(self.__experience(state, action, reward, next_state, done))

    def sample(self):

        experiences = random.sample(self.__memory, k=self.__batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.__device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.__device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.__device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.__device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.__device)
        indices = None
        weights = None

        dones_flatten = dones.view(dones.numel(), -1)
        rewards_flatten = rewards.view(rewards.numel(), -1)

        return states, actions, rewards_flatten, next_states, dones_flatten, indices, weights

    def __len__(self):
        return len(self.__memory)

class PrioritizedReplay(object):
    """
    Proportional Prioritization
    """
    def __init__(self, capacity, batch_size, device, gamma=0.99, n_step=1, alpha=0.6, beta_start = 0.4, beta_frames=200000, parallel_env=4):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1 #for beta calculation
        self.batch_size = batch_size
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

        self.n_step = n_step
        self.parallel_env = parallel_env
        self.n_step_buffer = [deque(maxlen=self.n_step) for i in range(parallel_env)]
        self.iter_ = 0
        self.gamma = gamma
        self.device = device

        #self.__experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def is_ready(self):
        return len(self) >= self.batch_size

    def calc_multistep_return(self,n_step_buffer):
        val = 0
        for idx in range(self.n_step):
            gammaIdx = self.gamma**idx
            val += gammaIdx * np.array(n_step_buffer[idx][2])
        
        return n_step_buffer[0][0], n_step_buffer[0][1], val, n_step_buffer[-1][3], n_step_buffer[-1][4]
    
    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.
        
        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent 
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def add(self, state, action, reward, next_state, done):
        if self.iter_ == self.parallel_env:
            self.iter_ = 0
        assert state.ndim == next_state.ndim
        #state      = np.expand_dims(state, 0)
        #next_state = np.expand_dims(next_state, 0)
        
        # n_step calc
        self.n_step_buffer[self.iter_].append((state, action, reward, next_state, done))
        if len(self.n_step_buffer[self.iter_]) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return(self.n_step_buffer[self.iter_])

        max_prio = self.priorities.max() if self.buffer else 1.0 # gives max priority if buffer is not empty else 1

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            # puts the new data on the position of the oldes since it circles via pos variable
            # since if len(buffer) == capacity -> pos == 0 -> oldest memory (at least for the first round?) 
            self.buffer[self.pos] = (state, action, reward, next_state, done) 
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity # lets the pos circle in the ranges of capacity if pos+1 > cap --> new posi = 0
        self.iter_ += 1

        
    def sample(self):
        N = len(self.buffer)
        if N == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
            
        # calc P = p^a/sum(p^a)
        probs  = prios ** self.alpha
        P = probs/probs.sum()
        
        #gets the indices depending on the probability p
        indices = np.random.choice(N, self.batch_size, p=P) 
        samples = [self.buffer[idx] for idx in indices]
        
        beta = self.beta_by_frame(self.frame)
        self.frame+=1
                
        #Compute importance-sampling weight
        weights  = (N * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max() 
        weights  = np.array(weights, dtype=np.float32) 
        
        states, actions, rewards, next_states, dones = zip(*samples) 

        states = torch.FloatTensor(np.concatenate(states)).to(self.device)
        next_states = torch.FloatTensor(np.float32(np.concatenate(next_states))).to(self.device)
        actions = torch.LongTensor(np.concatenate(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1) 
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        weights = torch.FloatTensor(weights).to(self.device)

        dones_flatten = dones.view(dones.numel(), -1)
        rewards_flatten = rewards.view(rewards.numel(), -1)

        return states, actions, rewards_flatten, next_states, dones_flatten, indices, weights
        # experiences = samples
        # states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        # actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        # rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        # next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        # dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        # weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # dones_flatten = dones.view(dones.numel(), -1)
        # rewards_flatten = rewards.view(rewards.numel(), -1)

        

        #return states, actions, rewards_flatten, next_states, dones_flatten, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio 

    def __len__(self):
        return len(self.buffer)
