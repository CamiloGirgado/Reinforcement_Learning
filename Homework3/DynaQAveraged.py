import numpy as np
import matplotlib.pyplot as plt
import random

# Maze size
ROWS = 6
COLS = 9

start = (2, 0)
goal = (0, 8)

# Walls
walls = [(1,2),(2,2),(3,2),(4,5),(0,7),(1,7),(2,7),(3,7)]

# Actions: right, left, down, up
actions = [(0,1),(0,-1),(1,0),(-1,0)]
n_actions = len(actions)

# Hyperparameters
alpha = 0.1
gamma = 0.95
epsilon = 0.1

episodes = 50
runs = 30


def step(state, action):
    r, c = state
    dr, dc = actions[action]

    nr = r + dr
    nc = c + dc

    if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
        return state, 0

    if (nr, nc) in walls:
        return state, 0

    next_state = (nr, nc)

    if next_state == goal:
        return next_state, 1

    return next_state, 0


def epsilon_greedy(Q, state):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)

    r, c = state
    return np.argmax(Q[r, c])


def dyna_q(planning_steps):

    Q = np.zeros((ROWS, COLS, n_actions))
    model = {}

    steps_per_episode = []

    for ep in range(episodes):

        state = start
        steps = 0

        while state != goal:

            action = epsilon_greedy(Q, state)

            next_state, reward = step(state, action)

            r, c = state
            nr, nc = next_state

            # Q-learning update
            Q[r, c, action] += alpha * (
                reward + gamma * np.max(Q[nr, nc]) - Q[r, c, action]
            )

            # store model
            model[(state, action)] = (next_state, reward)

            # planning updates
            for _ in range(planning_steps):

                s, a = random.choice(list(model.keys()))
                s_next, rwd = model[(s, a)]

                sr, sc = s
                snr, snc = s_next

                Q[sr, sc, a] += alpha * (
                    rwd + gamma * np.max(Q[snr, snc]) - Q[sr, sc, a]
                )

            state = next_state
            steps += 1

        steps_per_episode.append(steps)

    return np.array(steps_per_episode)


# Averaging across runs
results_n0 = np.zeros(episodes)
results_n5 = np.zeros(episodes)
results_n50 = np.zeros(episodes)

for run in range(runs):

    random.seed(run)
    np.random.seed(run)

    results_n0 += dyna_q(0)
    results_n5 += dyna_q(5)
    results_n50 += dyna_q(50)

results_n0 /= runs
results_n5 /= runs
results_n50 /= runs


# Plot results
plt.figure(figsize=(8,5))

plt.plot(results_n0, label="n = 0 (Q-learning)")
plt.plot(results_n5, label="n = 5")
plt.plot(results_n50, label="n = 50")

plt.xlabel("Episode")
plt.ylabel("Steps per Episode")
plt.title("Dyna-Q Maze (Averaged Over 20 Runs)")
plt.legend()
plt.grid()

plt.show()