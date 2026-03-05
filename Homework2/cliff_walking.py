import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Environment
ROWS, COLS = 4, 12
START = (3, 0)
GOAL  = (3, 11)
ACTIONS = [(-1,0),(1,0),(0,1),(0,-1)]  # up, down, right, left

def step(r, c, a):
    nr = np.clip(r + ACTIONS[a][0], 0, ROWS-1)
    nc = np.clip(c + ACTIONS[a][1], 0, COLS-1)
    if nr == 3 and 1 <= nc <= 10:          # cliff
        return START[0], START[1], -100, False
    if (nr, nc) == GOAL:
        return nr, nc, -1, True
    return nr, nc, -1, False

def epsilon_greedy(Q, r, c, eps):
    if np.random.rand() < eps:
        return np.random.randint(4)
    return np.argmax(Q[r, c])

def run_episode(Q, alpha, eps, method):
    r, c = START
    a = epsilon_greedy(Q, r, c, eps)
    total = 0
    for _ in range(10_000):
        nr, nc, reward, done = step(r, c, a)
        total += reward
        if method == "sarsa":
            na = epsilon_greedy(Q, nr, nc, eps)
            target = reward + (0 if done else Q[nr, nc, na])
            Q[r, c, a] += alpha * (target - Q[r, c, a])
            a = na
        else:  # Q-learning
            target = reward + (0 if done else np.max(Q[nr, nc]))
            Q[r, c, a] += alpha * (target - Q[r, c, a])
            a = epsilon_greedy(Q, nr, nc, eps)
        r, c = nr, nc
        if done:
            break
    return total

def extract_path(Q):
    path = []
    r, c = START
    visited = set()
    for _ in range(200):
        path.append((r, c))
        if (r, c) == GOAL:
            break
        if (r, c) in visited:
            break
        visited.add((r, c))
        nr, nc, _, _ = step(r, c, int(np.argmax(Q[r, c])))
        r, c = nr, nc
    return path

def smooth(x, w=15):
    return np.convolve(x, np.ones(w)/w, mode='same')

def train(episodes, alpha, eps_schedule, method):
    Q = np.zeros((ROWS, COLS, 4))
    rewards = []
    for ep in range(episodes):
        eps = eps_schedule(ep)
        rewards.append(run_episode(Q, alpha, eps, method))
    return Q, np.array(rewards)

# Draw Grid
def draw_grid(ax, sarsa_path, q_path, title):
    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11, color='white', pad=8)
    ax.set_facecolor('#0a0a0f')

    sarsa_set = set(sarsa_path)
    q_set     = set(q_path)

    for r in range(ROWS):
        for c in range(COLS):
            # cliff
            if r == 3 and 1 <= c <= 10:
                color = '#3d1212'
            elif (r, c) == GOAL:
                color = '#123d22'
            elif (r, c) == START:
                color = '#12123d'
            else:
                color = '#111118'
            rect = plt.Rectangle((c, ROWS-1-r), 1, 1, facecolor=color,
                                   edgecolor='#1a1a2e', linewidth=0.5)
            ax.add_patch(rect)
            if r == 3 and 1 <= c <= 10:
                ax.text(c+0.5, ROWS-1-r+0.5, 'C', ha='center', va='center',
                        color='#ff4444', fontsize=7, fontweight='bold')
            elif (r, c) == GOAL:
                ax.text(c+0.5, ROWS-1-r+0.5, 'G', ha='center', va='center',
                        color='#44ff88', fontsize=9, fontweight='bold')
            elif (r, c) == START:
                ax.text(c+0.5, ROWS-1-r+0.5, 'S', ha='center', va='center',
                        color='#8888ff', fontsize=9, fontweight='bold')

    # Draw paths as lines
    def plot_path(path, color, offset, label):
        if len(path) < 2:
            return
        xs = [c + 0.5 + offset for (r, c) in path]
        ys = [ROWS-1-r + 0.5 + offset for (r, c) in path]
        ax.plot(xs, ys, color=color, linewidth=2, alpha=0.85,
                marker='o', markersize=3, label=label)

    plot_path(sarsa_path, '#4dc8ff',  0.08, 'SARSA')
    plot_path(q_path,     '#ff4d4d', -0.08, 'Q-learning')
    ax.legend(loc='upper right', fontsize=8, facecolor='#111', labelcolor='white',
              framealpha=0.8)
    ax.set_xticks([])
    ax.set_yticks([])


# Main
def main():
    EPISODES = 500
    ALPHA    = 0.5
    EPS      = 0.1

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 11), facecolor='#07070d')
    fig.suptitle('Cliff Walking: SARSA vs Q-Learning', fontsize=14,
                 color='white', y=0.98)

    # Part 1 & 2: Fixed epsilon
    print("Training with fixed ε = 0.1")
    fixed_eps = lambda ep: EPS

    Qs_fixed, sr_fixed = train(EPISODES, ALPHA, fixed_eps, "sarsa")
    Qq_fixed, qr_fixed = train(EPISODES, ALPHA, fixed_eps, "qlearn")

    sarsa_path_fixed = extract_path(Qs_fixed)
    q_path_fixed     = extract_path(Qq_fixed)

    print(f"  SARSA path length:     {len(sarsa_path_fixed)} steps")
    print(f"  Q-learning path length:{len(q_path_fixed)} steps")
    print(f"  SARSA avg reward (last 50):     {sr_fixed[-50:].mean():.1f}")
    print(f"  Q-learning avg reward (last 50):{qr_fixed[-50:].mean():.1f}")

    # Part 3: Decaying epsilon 
    print("\nTraining with decaying ε (x0.99 per episode)")
    decay_eps = lambda ep: max(0.01, EPS * (0.99 ** ep))

    Qs_decay, sr_decay = train(EPISODES, ALPHA, decay_eps, "sarsa")
    Qq_decay, qr_decay = train(EPISODES, ALPHA, decay_eps, "qlearn")

    sarsa_path_decay = extract_path(Qs_decay)
    q_path_decay     = extract_path(Qq_decay)

    print(f"  SARSA path length:     {len(sarsa_path_decay)} steps")
    print(f"  Q-learning path length:{len(q_path_decay)} steps")
    print(f"  SARSA avg reward (last 50):     {sr_decay[-50:].mean():.1f}")
    print(f"  Q-learning avg reward (last 50):{qr_decay[-50:].mean():.1f}")

    # Plots 
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.25,
                          left=0.07, right=0.97, top=0.93, bottom=0.06)

    # Row 0: grids
    ax_grid1 = fig.add_subplot(gs[0, 0])
    ax_grid2 = fig.add_subplot(gs[0, 1])
    draw_grid(ax_grid1, sarsa_path_fixed, q_path_fixed,
              f'Learned Paths — Fixed ε={EPS}\n(SARSA safe, Q-learn optimal)')
    draw_grid(ax_grid2, sarsa_path_decay, q_path_decay,
              f'Learned Paths — Decaying ε')

    # Row 1: reward curves fixed eps
    ax_r1 = fig.add_subplot(gs[1, :])
    ep_x = np.arange(EPISODES)
    ax_r1.plot(ep_x, smooth(sr_fixed), color='#4dc8ff', lw=1.8, label='SARSA')
    ax_r1.plot(ep_x, smooth(qr_fixed), color='#ff4d4d', lw=1.8, label='Q-learning')
    ax_r1.axhline(y=-13, color='#aabb44', lw=1, ls='--', alpha=0.6, label='Optimal')
    ax_r1.set_title(f'Sum of Rewards per Episode — Fixed ε={EPS}  (smoothed)',
                    color='white', fontsize=10)
    ax_r1.set_xlabel('Episodes', color='#aaa')
    ax_r1.set_ylabel('Sum of Rewards', color='#aaa')
    ax_r1.set_facecolor('#0a0a0f')
    ax_r1.legend(fontsize=9, facecolor='#111', labelcolor='white')
    ax_r1.tick_params(colors='#666')
    for spine in ax_r1.spines.values():
        spine.set_edgecolor('#222')

    # Row 2: reward curves decay eps
    ax_r2 = fig.add_subplot(gs[2, :])
    ax_r2.plot(ep_x, smooth(sr_decay), color='#4dc8ff', lw=1.8, label='SARSA')
    ax_r2.plot(ep_x, smooth(qr_decay), color='#ff4d4d', lw=1.8, label='Q-learning')
    ax_r2.axhline(y=-13, color='#aabb44', lw=1, ls='--', alpha=0.6, label='Optimal')
    ax_r2.set_title('Sum of Rewards per Episode — Decaying ε  (smoothed)',
                    color='white', fontsize=10)
    ax_r2.set_xlabel('Episodes', color='#aaa')
    ax_r2.set_ylabel('Sum of Rewards', color='#aaa')
    ax_r2.set_facecolor('#0a0a0f')
    ax_r2.legend(fontsize=9, facecolor='#111', labelcolor='white')
    ax_r2.tick_params(colors='#666')
    for spine in ax_r2.spines.values():
        spine.set_edgecolor('#222')

    plt.show()


if __name__ == "__main__":
    main()