import torch.multiprocessing as mp
from actor_critic import ActorCritic
from optimizer import SharedAdam
import torch as T
from agent import Agent


lr = 1e-5
alpha = 0.95
env_id = 'CartPole-v1'
N_GAMES = 3000
T_MAX = 5
input_dims = [4]
n_actions = 2
distribution = T.distributions.Categorical
global_actor_critic = ActorCritic(input_dims, n_actions, distribution)
global_actor_critic.share_memory()
global_optimizer = SharedAdam(global_actor_critic.parameters(), lr=lr)
global_ep_idx = mp.Value('i', 0)

if __name__ == '__main__':
    n_threads = 8
    agents = [Agent(global_actor_critic, global_optimizer, input_dims, n_actions, distribution, 0.99, alpha, lr, i, global_ep_idx, env_id, N_GAMES, T_MAX) for i in range(n_threads)]
    [agent.start() for agent in agents]
    [agent.join() for agent in agents]
    print("Training complete")
