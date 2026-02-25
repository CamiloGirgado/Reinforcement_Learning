import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Environment definition
# ----------------------------------------------------------------------
rows = 4
cols = 12
start = (3, 0)          # bottom‑left
goal = (3, 11)          # bottom‑right
cliff = [(3, c) for c in range(1, 11)]   # bottom row except start and goal

# Actions: up, down, left, right
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
action_names = ['up', 'down', 'left', 'right']
num_actions = 4

def is_cliff(state):
    """Check if a state is part of the cliff."""
    return state in cliff

def step(state, action):
    """
    Take a step from the given state using the given action.
    Returns (next_state, reward, done).
    """
    r, c = state
    dr, dc = actions[action]
    nr, nc = r + dr, c + dc

    # Stay inside grid if action would move outside
    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
        nr, nc = r, c

    next_state = (nr, nc)

    if next_state == goal:
        return next_state, -1, True          # goal reached
    elif next_state in cliff:
        return start, -100, False            # fall into cliff, restart
    else:
        return next_state, -1, False         # normal move

# ----------------------------------------------------------------------
# Action selection
# ----------------------------------------------------------------------
def epsilon_greedy(Q, state, epsilon):
    """
    Choose an action using an epsilon‑greedy policy.
    Q is a 3D array of shape (rows, cols, num_actions).
    """
    if np.random.random() < epsilon:
        return np.random.choice(num_actions)
    else:
        return np.argmax(Q[state[0], state[1], :])

# ----------------------------------------------------------------------
# SARSA (on‑policy TD control)
# ----------------------------------------------------------------------
def sarsa(episodes, alpha=0.5, gamma=1.0, epsilon=0.1, decay=False):
    """
    Run SARSA for a given number of episodes.
    If decay=True, epsilon is exponentially decayed towards a minimum.
    Returns (Q_table, rewards_per_episode).
    """
    Q = np.zeros((rows, cols, num_actions))
    rewards_per_episode = []

    for ep in range(episodes):
        if decay:
            # Gradually reduce epsilon, but keep it >= 0.01
            epsilon = max(0.01, epsilon * 0.995)

        state = start
        # Choose first action using current policy
        action = epsilon_greedy(Q, state, epsilon)
        total_reward = 0
        done = False

        while not done:
            next_state, reward, done = step(state, action)
            # Choose next action (for the update and next step)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            # SARSA update
            Q[state[0], state[1], action] += alpha * (
                reward + gamma * Q[next_state[0], next_state[1], next_action]
                - Q[state[0], state[1], action]
            )

            state, action = next_state, next_action
            total_reward += reward

        rewards_per_episode.append(total_reward)

    return Q, rewards_per_episode

# ----------------------------------------------------------------------
# Q‑learning (off‑policy TD control)
# ----------------------------------------------------------------------
def q_learning(episodes, alpha=0.5, gamma=1.0, epsilon=0.1, decay=False):
    """
    Run Q‑learning for a given number of episodes.
    If decay=True, epsilon is exponentially decayed towards a minimum.
    Returns (Q_table, rewards_per_episode).
    """
    Q = np.zeros((rows, cols, num_actions))
    rewards_per_episode = []

    for ep in range(episodes):
        if decay:
            epsilon = max(0.01, epsilon * 0.995)

        state = start
        total_reward = 0
        done = False

        while not done:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done = step(state, action)

            # Q‑learning update: use max over next actions
            best_next = np.max(Q[next_state[0], next_state[1], :])
            Q[state[0], state[1], action] += alpha * (
                reward + gamma * best_next - Q[state[0], state[1], action]
            )

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

    return Q, rewards_per_episode

# ----------------------------------------------------------------------
# Visualisation helpers
# ----------------------------------------------------------------------
def extract_policy(Q):
    """Return a 2D array of greedy actions from the Q‑table."""
    return np.argmax(Q, axis=2)

def plot_path(policy, title):
    """
    Trace the greedy path from start to goal using the given policy.
    Display the path over the grid (cliff marked differently).
    """
    path = []
    state = start
    visited = set()
    # Follow greedy actions until goal or a loop (safety)
    while state != goal and state not in visited:
        visited.add(state)
        r, c = state
        a = policy[r, c]
        dr, dc = actions[a]
        nr, nc = r + dr, c + dc
        # Keep inside bounds (same as step function)
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
            nr, nc = r, c
        path.append((r, c))
        state = (nr, nc)
    path.append(goal)

    # Create a grid for display:
    # - cliff cells get -1 (shown as a distinct colour)
    # - start = 0.5, goal = 0.8, path = 1
    grid = np.zeros((rows, cols))
    for r, c in cliff:
        grid[r, c] = -1
    grid[start] = 0.5
    grid[goal] = 0.8
    for (r, c) in path:
        grid[r, c] = 1

    plt.figure(figsize=(10, 3))
    plt.imshow(grid, cmap='viridis', origin='upper', vmin=-1, vmax=1)
    plt.colorbar(ticks=[-1, 0, 0.5, 0.8, 1], label='Value')
    plt.title(title)
    plt.xticks(range(cols))
    plt.yticks(range(rows))
    plt.show()

# ----------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------
if __name__ == "__main__":
    episodes = 500

    # ---------- Part 1: Fixed epsilon = 0.1 ----------
    print("Running SARSA with ε = 0.1 ...")
    Q_sarsa, rewards_sarsa = sarsa(episodes, epsilon=0.1)
    print("Running Q‑learning with ε = 0.1 ...")
    Q_q, rewards_q = q_learning(episodes, epsilon=0.1)

    # Plot learning curves
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_sarsa, label='SARSA', alpha=0.7)
    plt.plot(rewards_q, label='Q‑learning', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Sum of rewards per episode')
    plt.legend()
    plt.title('Cliff Walking: SARSA vs Q‑learning (ε = 0.1)')
    plt.grid(True)
    plt.show()

    # Extract and display greedy policies
    policy_sarsa = extract_policy(Q_sarsa)
    policy_q = extract_policy(Q_q)
    plot_path(policy_sarsa, 'SARSA greedy path (safe)')
    plot_path(policy_q, 'Q‑learning greedy path (optimal)')

    # ---------- Part 2: With epsilon decay ----------
    print("\nRunning SARSA with ε decay ...")
    Q_sarsa_dec, rewards_sarsa_dec = sarsa(episodes, epsilon=0.1, decay=True)
    print("Running Q‑learning with ε decay ...")
    Q_q_dec, rewards_q_dec = q_learning(episodes, epsilon=0.1, decay=True)

    # Plot learning curves with decay
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_sarsa_dec, label='SARSA (decay)', alpha=0.7)
    plt.plot(rewards_q_dec, label='Q‑learning (decay)', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Sum of rewards per episode')
    plt.legend()
    plt.title('Cliff Walking: SARSA vs Q‑learning (ε decays from 0.1 to 0.01)')
    plt.grid(True)
    plt.show()

    # Greedy policies after decay
    policy_sarsa_dec = extract_policy(Q_sarsa_dec)
    policy_q_dec = extract_policy(Q_q_dec)
    plot_path(policy_sarsa_dec, 'SARSA with ε decay (now optimal)')
    plot_path(policy_q_dec, 'Q‑learning with ε decay (optimal)')