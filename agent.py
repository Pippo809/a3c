import torch as T
from torch.distributions.distribution import Distribution
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from actor_critic import ActorCritic
import torch.multiprocessing as mp
import gym


class Agent(mp.Process):
    def __init__(self, global_ac, optimizer, input_dims, n_actions, distribution,
                 gamma, alpha, lr, name, global_ep_idx, env_id, N_GAMES=5000, T_MAX=5):
        super(Agent, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.local_ac = ActorCritic(input_dims, n_actions, distribution, gamma, alpha)
        self.global_ac = global_ac
        self.name = str(name)
        self.episode_idx = global_ep_idx
        self.env = gym.make(env_id)
        print(self.name, "env", self.env)
        print("Env spaces", self.env.observation_space, self.env.action_space)
        self.gamma = gamma
        self.optimizer = optimizer
        self.N_GAMES = N_GAMES
        self.T_MAX = T_MAX
    
    def run(self):
        t_step = 1
        while self.episode_idx.value < self.N_GAMES:
            done = False
            obs, _ = self.env.reset()
            score = 0
            self.local_ac.clear_memory()
            while not done:
                action = self.local_ac.choose_action(obs)
                obs_, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                score += reward
                self.local_ac.remember(obs, action, reward)
                if t_step % self.T_MAX == 0 or done:
                    loss = self.local_ac.calc_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(
                        self.local_ac.parameters(),
                        self.global_ac.parameters()
                    ):
                        global_param._grad = local_param.grad
                    self.optimizer.step()
                    self.local_ac.load_state_dict(
                        self.global_ac.state_dict()
                    )
                    self.local_ac.clear_memory()
                t_step += 1
                obs = obs_
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            print(self.name, "episode ", self.episode_idx.value, f"Reward {score:.2f}")
