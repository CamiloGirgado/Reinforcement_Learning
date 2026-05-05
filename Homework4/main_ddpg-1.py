import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

class ReplayBuffer:
    def __init__(self, max_size, state_dim, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, state_dim))
        self.new_state_memory = np.zeros((self.mem_size, state_dim))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        return self.state_memory[batch], self.action_memory[batch], \
               self.reward_memory[batch], self.new_state_memory[batch], \
               self.terminal_memory[batch]

class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dim, n_actions):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.mu = nn.Linear(300, n_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.mu(x)) # Bound actions to [-1, 1][cite: 1]

class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dim, n_actions):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + n_actions, 400)
        self.fc2 = nn.Linear(400, 300)
        self.q = nn.Linear(300, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        return self.q(x)

class DDPGAgent:
    def __init__(self, state_dim, n_actions, alpha=0.0001, beta=0.001, gamma=0.99, tau=0.005):
        self.gamma, self.tau = gamma, tau
        self.actor = ActorNetwork(alpha, state_dim, n_actions)
        self.critic = CriticNetwork(beta, state_dim, n_actions)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

    def choose_action(self, observation):
        self.actor.eval()
        # Convert to numpy array first to avoid the PyTorch performance warning
        state = torch.tensor(np.array([observation]), dtype=torch.float).to(self.actor.device)
        mu = self.actor(state).detach()
        noise = torch.randn_like(mu) * 0.1 
        action = (mu + noise).cpu().numpy()[0]
        return np.clip(action, -1, 1)

    def learn(self, memory, batch_size):
        if memory.mem_cntr < batch_size: return
        s, a, r, s_, d = memory.sample_buffer(batch_size)
        s, a, r, s_, d = [torch.tensor(x, dtype=torch.float).to(self.actor.device) for x in [s, a, r, s_, d]]

        # Critic update
        with torch.no_grad():
            target_actions = self.target_actor(s_)
            q_next = self.target_critic(s_, target_actions).flatten()
            q_next[d.bool()] = 0.0
            target = r + self.gamma * q_next
        q_eval = self.critic(s, a).flatten()
        critic_loss = F.mse_loss(target, q_eval)
        self.critic.optimizer.zero_grad(); critic_loss.backward(); self.critic.optimizer.step()

        # Actor update
        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor.optimizer.zero_grad(); actor_loss.backward(); self.actor.optimizer.step()

        # Soft update[cite: 1]
        for target, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)
        for target, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)

if __name__ == '__main__':
    from copy import deepcopy
    env = gym.make('LunarLanderContinuous-v3')
    agent = DDPGAgent(env.observation_space.shape[0], env.action_space.shape[0])
    memory = ReplayBuffer(1000000, env.observation_space.shape[0], env.action_space.shape[0])
    writer = SummaryWriter(log_dir="./logs/ddpg")
    for i in range(1000):
        obs, info = env.reset() # This extracts just the array into 'obs'
        score = 0
        done = False
        while not done:
            action = agent.choose_action(obs)
            # Unpack the 5 values: observation, reward, terminated, truncated, info
            obs_, reward, terminated, truncated, info = env.step(action)

            # Combine terminated and truncated for your ReplayBuffer logic
            done = terminated or truncated
            memory.store_transition(obs, action, reward, obs_, done)
            agent.learn(memory, 64)
            obs, score = obs_, score + reward
        writer.add_scalar("Reward", score, i)
        print(f"Episode: {i}, Score: {score}")