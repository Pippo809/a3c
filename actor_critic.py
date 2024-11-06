import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, distribution: type[T.distributions.Distribution], gamma=0.99, alpha=0.0003):
        super(ActorCritic, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.distribution = distribution
        
        self.p1 = nn.Linear(*self.input_dims, 256)
        self.v1 = nn.Linear(*self.input_dims, 256)
        
        self.pi = nn.Linear(256, self.n_actions)
        self.v = nn.Linear(256, 1)
        
        self.clear_memory()
        
    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def forward(self, state):
        pi = F.relu(self.p1(state))
        vi = F.relu(self.p1(state))
        
        pi = self.pi(pi)
        v = self.v(vi)
        return pi, v
    
    def calc_R(self, done):
        states = T.tensor(self.states, dtype=T.float)
        _, v = self.forward(states)
        
        R = v[-1]*(1-int(done))  # Terminal
        
        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma*R
            batch_return.append(R)
        batch_return.reverse()
        
        return T.tensor(batch_return, dtype=T.float)
    
    def calc_loss(self, done):
        
        states = T.tensor(self.states, dtype=T.float)
        actions = T.tensor(self.actions, dtype=T.float)
        
        returns = self.calc_R(done)
        
        pi, values = self.forward(states)
        values = values.squeeze()
        
        critic_loss = (returns-values)**2
        
        probs = T.softmax(pi, dim=1)
        dist = self.distribution(probs)
        log_prob = dist.log_prob(actions)
        
        actor_loss = -log_prob*(returns-values)
        total_loss = (critic_loss+actor_loss).mean()
        return total_loss
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        
    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).unsqueeze(0)
        pi, _ = self.forward(state)
        action = self.distribution(logits=pi).sample()
        
        return action.cpu().numpy()[0]