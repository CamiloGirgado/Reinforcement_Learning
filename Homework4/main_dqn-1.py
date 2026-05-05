import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

class EpsilonDecayScheduler:
    def __init__(self, eps_start, eps_finish, eps_aneal_time):
        self.eps_start = eps_start
        self.eps_finish = eps_finish
        self.eps_aneal_time = eps_aneal_time
        self.dd = (self.eps_start - self.eps_finish) / self.eps_aneal_time

    def get_epsilon(self, current_time_step):
        return max(self.eps_finish, self.eps_start - self.dd * current_time_step)

class DeepQNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = state.to(torch.float32)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, gamma, epsilon_start, lr, input_dims, batch_size, n_actions, 
                 epsilon_aneal_time, replay_buffer_size, epsilon_end):
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_actions)]
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer_couter = 0
        
        self.Q_eval = DeepQNetwork(input_dims, 256, 256, n_actions)
        self.Q_target = deepcopy(self.Q_eval) # Independent target network[cite: 2]
        self.optimizer = optim.Adam(self.Q_eval.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.epsilon_decay_scheduler = EpsilonDecayScheduler(epsilon_start, epsilon_end, epsilon_aneal_time)
        self.state_memory = np.zeros((self.replay_buffer_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.replay_buffer_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.replay_buffer_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.replay_buffer_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.replay_buffer_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.replay_buffer_couter % self.replay_buffer_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal
        self.replay_buffer_couter += 1

    def choose_action(self, observation, curr_env_step):
        self.epsilon = self.epsilon_decay_scheduler.get_epsilon(curr_env_step)
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            return torch.argmax(actions).item()
        return np.random.choice(self.action_space)

    def learn(self):
        if self.replay_buffer_couter < self.batch_size:
            return
        self.optimizer.zero_grad()
        max_mem = min(self.replay_buffer_couter, self.replay_buffer_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        
        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[np.arange(self.batch_size), action_batch]
        with torch.no_grad():
            q_next = self.Q_target.forward(new_state_batch)
            q_next[terminal_batch] = 0.0
        
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        loss = self.loss(q_target, q_eval)
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    env = gym.make('LunarLander-v3')
    agent = Agent(gamma=0.99, epsilon_start=1.0, lr=0.001, input_dims=[8], 
                  batch_size=64, n_actions=4, epsilon_aneal_time=100000, 
                  epsilon_end=0.01, replay_buffer_size=50000)
    writer = SummaryWriter(log_dir="./logs/dqn")
    total_steps = 0
    for i in range(1000):
        obs, info = env.reset()
        score, done = 0, False
        while not done:
            action = agent.choose_action(obs, total_steps)
            obs_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(obs, action, reward, obs_, done)
            agent.learn()
            obs, score, total_steps = obs_, score + reward, total_steps + 1
        if i % 10 == 0:
            agent.Q_target.load_state_dict(agent.Q_eval.state_dict())
        writer.add_scalar("Reward", score, i)
        print(f"Episode: {i}, Score: {score}")