import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Environment
# ===============================
ROWS, COLS = 4, 12
START = (3, 0)
GOAL = (3, 11)
CLIFF = [(3, c) for c in range(1, 11)]

ACTIONS = [0, 1, 2, 3]  # up, right, down, left
ACTION_DELTA = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1)
}

ACTION_SYMBOL = {
    0: "↑",
    1: "→",
    2: "↓",
    3: "←"
}


def step(state, action):
    r, c = state
    dr, dc = ACTION_DELTA[action]
    nr, nc = r + dr, c + dc
    nr = np.clip(nr, 0, ROWS - 1)
    nc = np.clip(nc, 0, COLS - 1)
    next_state = (nr, nc)

    if next_state in CLIFF:
        return START, -100, False
    if next_state == GOAL:
        return next_state, 0, True

    return next_state, -1, False


# ===============================
# ε-greedy
# ===============================
def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(ACTIONS)
    return np.argmax(Q[state])


# ===============================
# SARSA
# ===============================
def sarsa(episodes, alpha=0.5, epsilon=0.1, decay=None):
    Q = np.zeros((ROWS, COLS, 4))
    rewards = []

    for ep in range(episodes):
        state = START
        action = epsilon_greedy(Q, state, epsilon)
        total_reward = 0
        done = False

        while not done:
            next_state, reward, done = step(state, action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            Q[state][action] += alpha * (
                reward + Q[next_state][next_action] - Q[state][action]
            )

            state, action = next_state, next_action
            total_reward += reward

        rewards.append(total_reward)
        if decay:
            epsilon = max(0.01, epsilon * decay)

    return Q, rewards


# ===============================
# Q-learning
# ===============================
def q_learning(episodes, alpha=0.5, epsilon=0.1, decay=None):
    Q = np.zeros((ROWS, COLS, 4))
    rewards = []

    for ep in range(episodes):
        state = START
        total_reward = 0
        done = False

        while not done:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done = step(state, action)

            Q[state][action] += alpha * (
                reward + np.max(Q[next_state]) - Q[state][action]
            )

            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        if decay:
            epsilon = max(0.01, epsilon * decay)

    return Q, rewards


# ===============================
# Policy visualization
# ===============================
def plot_policy(Q, title):
    grid = [[" " for _ in range(COLS)] for _ in range(ROWS)]

    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) == START:
                grid[r][c] = "S"
            elif (r, c) == GOAL:
                grid[r][c] = "G"
            elif (r, c) in CLIFF:
                grid[r][c] = "X"
            else:
                grid[r][c] = ACTION_SYMBOL[np.argmax(Q[r, c])]

    print("\n" + title)
    for row in grid:
        print(" ".join(row))


# ===============================
# Run experiment
# ===============================
if __name__ == "__main__":
    EPISODES = 500
    ALPHA = 0.5
    EPSILON = 0.1

    # ---- Fixed epsilon (book result) ----
    Q_sarsa, R_sarsa = sarsa(EPISODES, ALPHA, EPSILON)
    Q_ql, R_ql = q_learning(EPISODES, ALPHA, EPSILON)

    plt.figure(figsize=(8, 5))
    plt.plot(R_sarsa, label="SARSA")
    plt.plot(R_ql, label="Q-learning")
    plt.xlabel("Episodes")
    plt.ylabel("Sum of rewards per episode")
    plt.title("Cliff Walking (ε = 0.1)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plot_policy(Q_sarsa, "SARSA Policy (Safe Path)")
    plot_policy(Q_ql, "Q-learning Policy (Optimal Path)")

    # ---- Epsilon decay ----
    Q_sarsa_d, R_sarsa_d = sarsa(EPISODES, ALPHA, EPSILON, decay=0.995)
    Q_ql_d, R_ql_d = q_learning(EPISODES, ALPHA, EPSILON, decay=0.995)

    plt.figure(figsize=(8, 5))
    plt.plot(R_sarsa_d, label="SARSA (ε decay)")
    plt.plot(R_ql_d, label="Q-learning (ε decay)")
    plt.xlabel("Episodes")
    plt.ylabel("Sum of rewards per episode")
    plt.title("Cliff Walking with ε Decay")
    plt.legend()
    plt.grid(True)
    plt.show()

    plot_policy(Q_sarsa_d, "SARSA Policy with ε Decay")
    plot_policy(Q_ql_d, "Q-learning Policy with ε Decay")