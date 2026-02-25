import numpy as np
import matplotlib.pyplot as plt

# --- Environment Constants ---
WIDTH = 12
HEIGHT = 4
START = (3, 0)
GOAL = (3, 11)
CLIFF = [(3, i) for i in range(1, 11)]

# Actions: 0=Up, 1=Down, 2=Left, 3=Right
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def step(state, action_idx):
    """Moves the agent and returns (next_state, reward)"""
    res_r = state[0] + ACTIONS[action_idx][0]
    res_c = state[1] + ACTIONS[action_idx][1]
    
    # Keep agent within grid boundaries
    res_r = max(0, min(HEIGHT - 1, res_r))
    res_c = max(0, min(WIDTH - 1, res_c))
    
    new_state = (res_r, res_c)
    
    if new_state in CLIFF:
        return START, -100
    if new_state == GOAL:
        return new_state, 0
    return new_state, -1

def epsilon_greedy(state, q_table, epsilon):
    """Returns an action index using epsilon-greedy logic"""
    if np.random.rand() < epsilon:
        return np.random.choice(len(ACTIONS))
    return np.argmax(q_table[state])

def run_experiment(method='sarsa', episodes=500, alpha=0.5, epsilon=0.1, gamma=1.0, decay=False):
    """Runs the training loop for a specific RL method"""
    q_table = np.zeros((HEIGHT, WIDTH, len(ACTIONS)))
    rewards_cache = []
    curr_epsilon = epsilon

    for ep in range(episodes):
        state = START
        total_reward = 0
        
        # Initial action selection
        action = epsilon_greedy(state, q_table, curr_epsilon)
        
        while state != GOAL:
            next_state, reward = step(state, action)
            total_reward += reward
            
            if method == 'sarsa':
                # On-policy: choose next action BEFORE updating
                next_action = epsilon_greedy(next_state, q_table, curr_epsilon)
                td_target = reward + gamma * q_table[next_state][next_action]
                q_table[state][action] += alpha * (td_target - q_table[state][action])
                state, action = next_state, next_action
                
            else: # q_learning
                # Off-policy: use MAX of next state regardless of actual next action
                best_next_val = np.max(q_table[next_state])
                td_target = reward + gamma * best_next_val
                q_table[state][action] += alpha * (td_target - q_table[state][action])
                state = next_state
                action = epsilon_greedy(state, q_table, curr_epsilon)
        
        rewards_cache.append(total_reward)
        
        if decay:
            curr_epsilon = max(0.001, curr_epsilon * 0.99)
            
    return rewards_cache, q_table

def print_policy(q_table, label):
    """Prints a visual grid of the learned policy"""
    symbols = ['↑', '↓', '←', '→']
    print(f"\n--- Learned Policy: {label} ---")
    for r in range(HEIGHT):
        row = []
        for c in range(WIDTH):
            if (r, c) == GOAL: row.append(" G ")
            elif (r, c) in CLIFF: row.append(" C ")
            else:
                best_action = np.argmax(q_table[r, c])
                row.append(f" {symbols[best_action]} ")
        print("".join(row))

# --- Run Standard Comparison (Fixed Epsilon) ---
sarsa_rewards, sarsa_q = run_experiment(method='sarsa')
ql_rewards, ql_q = run_experiment(method='q_learning')

# --- Run Epsilon Decay Comparison ---
sarsa_decay_rewards, sarsa_decay_q = run_experiment(method='sarsa', decay=True)
ql_decay_rewards, ql_decay_q = run_experiment(method='q_learning', decay=True)

# --- Visualization ---
plt.figure(figsize=(10, 6))
plt.plot(sarsa_rewards, label='Sarsa (ε=0.1)', alpha=0.7)
plt.plot(ql_rewards, label='Q-learning (ε=0.1)', alpha=0.7)
plt.title('Sarsa vs Q-Learning: Cliff Walking Rewards')
plt.xlabel('Episodes')
plt.ylabel('Sum of rewards')
plt.ylim(-100, 0)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Show the learned paths
print_policy(sarsa_q, "Sarsa (Fixed Epsilon)")
print_policy(ql_q, "Q-Learning (Fixed Epsilon)")
print_policy(sarsa_decay_q, "Sarsa (Epsilon Decay)")