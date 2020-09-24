import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
E_p = 0.01       # to affect zero priority
A_p = 0.6        # affects degree of priority sampling 1->pure priority sampling 0->uniform sampling

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def WEightedMSE(input, target, weights = 1):
    return torch.sum(weights*(input-target)**2)

class Agent():
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)


        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)


        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        self.t_step = 0

    def step(self, state, action, reward, next_state, done, priority, B_P):

        self.memory.add(state, action, reward, next_state, done, priority)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(B_P)
                self.learn(experiences, GAMMA)

    def priority(self, states, actions, rewards, next_states):
        if len(states.shape) == 1: 
            # only a single experience tuple to evaluate
            # need to format variables accordingly:
            states = torch.from_numpy(states).float().unsqueeze(0).to(device)
            next_states = torch.from_numpy(next_states).float().unsqueeze(0).to(device)
            rewards = torch.tensor([[rewards]], dtype=torch.float).to(device) # scalar value
            actions = torch.tensor([[actions]], dtype=torch.uint8).to(device) # scalar value

        action_local = self.qnetwork_local.forward(next_states).argmax(1)
        max_q = self.qnetwork_target.forward(next_states)[np.arange(action_local.shape[0]), action_local]
        delta = (rewards.squeeze() + GAMMA*max_q) - self.qnetwork_local(states)[np.arange(actions.shape[0]),actions.byte().squeeze().cpu().numpy()]
        priority = torch.abs(delta) + E_p
        return priority.squeeze().tolist()

    def act(self, state, eps=0.):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()


        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones, weights, experience_indices = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))


        Q_expected = self.qnetwork_local(states).gather(1, actions)


        loss = WEightedMSE(Q_expected, Q_targets, weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        # ------------------- update priorities in the replay buffer ------------------- #
        new_priorities = self.priority(states, actions, rewards, next_states)        
        for count, idx in enumerate(experience_indices):
            self.memory.memory[idx] = self.memory.memory[idx]._replace(priority=new_priorities[count])

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done, priority):
        e = self.experience(state, action, reward, next_state, done, priority)
        self.memory.append(e)
    
    def sample(self, B_p):

        p = np.array([self.memory[i].priority for i in range(len(self.memory))])**A_p
        p /= p.sum()

        experience_indices = np.random.choice(len(self.memory), size=self.batch_size, replace=False, p=p)
        experiences = [self.memory[i] for i in experience_indices]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy((len(self.memory)*p[experience_indices])**(-B_p)).float().to(device)

        return (states, actions, rewards, next_states, dones, weights, experience_indices)

    def __len__(self):
        return len(self.memory)