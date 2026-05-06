import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class DQNReplayBuffer:
    def __init__(self, max_size, state_dim):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, state_dim), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, state_dim), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
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

class QNetwork(nn.Module):
    def __init__(self, lr, state_dim, n_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.q = nn.Linear(64, n_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.q(x)

class DQNAgent:
    def __init__(self, state_dim, n_actions, lr=0.001, gamma=0.99, epsilon=1.0, eps_dec=1e-4, eps_min=0.01, tau=0.005):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.tau = tau
        self.n_actions = n_actions
        
        self.q_eval = QNetwork(lr, state_dim, n_actions)
        self.q_target = QNetwork(lr, state_dim, n_actions)
        self.q_target.load_state_dict(self.q_eval.state_dict())

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor(np.array([observation]), dtype=torch.float).to(self.q_eval.device)
            actions = self.q_eval(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.n_actions)
        return action

    def learn(self, memory, batch_size):
        if memory.mem_cntr < batch_size:
            return
        
        s, a, r, s_, d = memory.sample_buffer(batch_size)
        s = torch.tensor(s).to(self.q_eval.device)
        a = torch.tensor(a).to(self.q_eval.device)
        r = torch.tensor(r).to(self.q_eval.device)
        s_ = torch.tensor(s_).to(self.q_eval.device)
        d = torch.tensor(d).to(self.q_eval.device)

        q_evals = self.q_eval(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.q_target(s_).max(1)[0]
            q_next[d] = 0.0
            target = r + self.gamma * q_next

        loss = F.mse_loss(q_evals, target)
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()

        # Soft update target network
        for target_param, eval_param in zip(self.q_target.parameters(), self.q_eval.parameters()):
            target_param.data.copy_(self.tau * eval_param.data + (1.0 - self.tau) * target_param.data)

        # Decay epsilon
        self.epsilon = max(self.eps_min, self.epsilon - self.eps_dec)


if __name__ == '__main__':
    env = gym.make('LunarLander-v3')
    agent = DQNAgent(state_dim=env.observation_space.shape[0], n_actions=env.action_space.n)
    memory = DQNReplayBuffer(100000, env.observation_space.shape[0])
    writer = SummaryWriter(log_dir="./logs/dqn")
    
    n_episodes = 1000
    score_history = []

    for i in range(n_episodes):
        obs, info = env.reset()
        score = 0
        done = False
        
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            memory.store_transition(obs, action, reward, obs_, done)
            agent.learn(memory, 64)
            
            obs = obs_
            score += reward
            
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        writer.add_scalar("Reward", score, i)
        print(f"Episode: {i}, Score: {score:.2f}, Avg Score (100 ep): {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")

    # Plotting Performance
    plt.figure(figsize=(10, 6))
    running_avg = np.zeros(len(score_history))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(score_history[max(0, i-100):(i+1)])
        
    plt.plot(score_history, alpha=0.3, label='Raw Episode Score')
    plt.plot(running_avg, color='red', linewidth=2, label='100-Episode Moving Avg')
    plt.axhline(y=100, color='green', linestyle='--', label='Success Target (>= 100)')
    plt.title('DQN Learning Curve - LunarLander-v3')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.savefig('dqn_learning_curve.png')
    plt.show()