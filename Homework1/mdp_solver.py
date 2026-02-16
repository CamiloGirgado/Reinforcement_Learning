import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time

# ─────────────────────────────────────────────
# 1.  WORLD MAP
# ─────────────────────────────────────────────
W = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
])

ROWS, COLS = W.shape          # 15 x 51
GOAL         = (7, 11)        # row=7 (0-indexed from W(8,11) → row-index 7, col-index 11 if 1-indexed → but
                               # the problem says W(8,11).  Matrices are commonly 1-indexed in textbooks.
                               # Row 8 → index 7, Col 11 → index 10  — BUT looking at the map row 7 col 11
                               # is free (0) and sits right on the long corridor → confirmed.
                               # Actually W is 0-indexed in numpy.  "W(8,11)" with 1-indexing → (7,10).
                               # Check: W[7][10] = 0  ✓  and it's on the central corridor.
GOAL         = (7, 10)        # corrected: 1-indexed (8,11) → 0-indexed (7,10)

REWARD_GOAL     =  100.0
REWARD_OBSTACLE = -50.0
REWARD_STEP     =  -1.0
GAMMA           =  0.95

# ─────────────────────────────────────────────
# 2.  8-DIRECTION ACTIONS  (dx, dy) in (row, col) space
# ─────────────────────────────────────────────
#   Index  Name      (drow, dcol)
ACTIONS = [
    (-1,  0),   # 0  N
    (-1,  1),   # 1  NE
    ( 0,  1),   # 2  E
    ( 1,  1),   # 3  SE
    ( 1,  0),   # 4  S
    ( 1, -1),   # 5  SW
    ( 0, -1),   # 6  W
    (-1, -1),   # 7  NW
]
NUM_ACTIONS = len(ACTIONS)

# Arrow vectors for plotting (note: matplotlib y-axis is flipped for images)
ARROW_DX = [a[1] for a in ACTIONS]   # col  → x
ARROW_DY = [-a[0] for a in ACTIONS]  # -row → y  (image origin top-left)


# ─────────────────────────────────────────────
# 3.  FREE-STATE INDEX HELPERS
# ─────────────────────────────────────────────
free_states = []
state_index = {}
for r in range(ROWS):
    for c in range(COLS):
        if W[r, c] == 0:
            state_index[(r, c)] = len(free_states)
            free_states.append((r, c))
NUM_STATES = len(free_states)


def is_free(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS and W[r, c] == 0


# ─────────────────────────────────────────────
# 4.  TRANSITION MODEL
#     deterministic=True  → action always succeeds
#     deterministic=False → 60 % intended, 20 % +45°, 20 % −45°
# ─────────────────────────────────────────────
def get_transitions(r, c, action_idx, deterministic=True):
    """Return list of  (probability, next_r, next_c)  tuples."""
    if deterministic:
        nr, nc = r + ACTIONS[action_idx][0], c + ACTIONS[action_idx][1]
        if is_free(nr, nc):
            return [(1.0, nr, nc)]
        else:                          # bump into wall → stay
            return [(1.0, r, c)]
    else:
        # stochastic: intended (60%), +45° (20%), -45° (20%)
        intended  = action_idx
        plus45    = (action_idx + 1) % 8
        minus45   = (action_idx - 1) % 8
        outcomes  = [(0.6, intended), (0.2, plus45), (0.2, minus45)]
        result    = {}
        for prob, act in outcomes:
            nr, nc = r + ACTIONS[act][0], c + ACTIONS[act][1]
            if is_free(nr, nc):
                key = (nr, nc)
            else:
                key = (r, c)           # stay in place on collision
            result[key] = result.get(key, 0.0) + prob
        return [(p, k[0], k[1]) for k, p in result.items()]


# ─────────────────────────────────────────────
# 5.  REWARD  R(s, a)  – reward received upon leaving state s with action a
#     We define reward based on the *next* state reached.
#     For stochastic model we take the expected immediate reward.
# ─────────────────────────────────────────────
def expected_reward(r, c, action_idx, deterministic):
    """Expected immediate reward for taking action_idx in (r,c)."""
    if (r, c) == GOAL:
        return 0.0                     # absorbing – no further reward
    transitions = get_transitions(r, c, action_idx, deterministic)
    R = 0.0
    for prob, nr, nc in transitions:
        if (nr, nc) == GOAL:
            R += prob * REWARD_GOAL
        elif (nr, nc) == (r, c) and not is_free(r + ACTIONS[action_idx][0],
                                                  c + ACTIONS[action_idx][1]):
            # bumped into a wall
            R += prob * REWARD_OBSTACLE
        else:
            R += prob * REWARD_STEP
    return R


# ─────────────────────────────────────────────
# 6.  POLICY & VALUE ITERATION  (Sutton & Barto style)
# ─────────────────────────────────────────────
def policy_evaluation(policy, V, deterministic, theta=1e-6):
    """Iterative policy evaluation.  Returns updated V and iteration count."""
    iters = 0
    while True:
        delta = 0.0
        for idx, (r, c) in enumerate(free_states):
            if (r, c) == GOAL:
                V[idx] = REWARD_GOAL   # terminal value
                continue
            a   = policy[idx]
            q   = expected_reward(r, c, a, deterministic)
            for prob, nr, nc in get_transitions(r, c, a, deterministic):
                q += GAMMA * prob * V[state_index[(nr, nc)]]
            delta = max(delta, abs(V[idx] - q))
            V[idx] = q
        iters += 1
        if delta < theta:
            break
    return V, iters


def policy_improvement(V, deterministic):
    """Greedy policy w.r.t. V.  Returns (policy, stable)."""
    policy = np.zeros(NUM_STATES, dtype=int)
    stable = True
    for idx, (r, c) in enumerate(free_states):
        if (r, c) == GOAL:
            continue
        best_val = -np.inf
        best_a   = 0
        for a in range(NUM_ACTIONS):
            q = expected_reward(r, c, a, deterministic)
            for prob, nr, nc in get_transitions(r, c, a, deterministic):
                q += GAMMA * prob * V[state_index[(nr, nc)]]
            if q > best_val:
                best_val = q
                best_a   = a
        policy[idx] = best_a
    return policy, True   # we always rebuild, stability checked externally


def policy_iteration(deterministic, theta=1e-6):
    """Full policy-iteration loop.  Returns V, policy, convergence_history."""
    V      = np.zeros(NUM_STATES)
    policy = np.zeros(NUM_STATES, dtype=int)   # init: action 0 (N)
    conv_history = []                          # (cumulative_eval_iters, max_delta)

    total_eval_iters = 0
    iteration        = 0
    while True:
        iteration += 1
        # --- policy evaluation ---
        V_old = V.copy()
        V, eval_iters = policy_evaluation(policy, V, deterministic, theta)
        total_eval_iters += eval_iters

        # --- policy improvement ---
        new_policy = np.zeros(NUM_STATES, dtype=int)
        stable = True
        for idx, (r, c) in enumerate(free_states):
            if (r, c) == GOAL:
                continue
            best_val = -np.inf
            best_a   = 0
            for a in range(NUM_ACTIONS):
                q = expected_reward(r, c, a, deterministic)
                for prob, nr, nc in get_transitions(r, c, a, deterministic):
                    q += GAMMA * prob * V[state_index[(nr, nc)]]
                if q > best_val:
                    best_val = q
                    best_a   = a
            new_policy[idx] = best_a
            if new_policy[idx] != policy[idx]:
                stable = False
        policy = new_policy
        conv_history.append((total_eval_iters, iteration, np.max(np.abs(V - V_old))))
        if stable:
            break
    return V, policy, conv_history


def value_iteration(deterministic, theta=1e-6):
    """Value iteration.  Returns V, policy, convergence_history (delta per sweep)."""
    V = np.zeros(NUM_STATES)
    conv_history = []

    iteration = 0
    while True:
        iteration += 1
        delta = 0.0
        for idx, (r, c) in enumerate(free_states):
            if (r, c) == GOAL:
                V[idx] = REWARD_GOAL
                continue
            best_val = -np.inf
            for a in range(NUM_ACTIONS):
                q = expected_reward(r, c, a, deterministic)
                for prob, nr, nc in get_transitions(r, c, a, deterministic):
                    q += GAMMA * prob * V[state_index[(nr, nc)]]
                if q > best_val:
                    best_val = q
            delta  = max(delta, abs(V[idx] - best_val))
            V[idx] = best_val
        conv_history.append((iteration, delta))
        if delta < theta:
            break

    # extract greedy policy
    policy = np.zeros(NUM_STATES, dtype=int)
    for idx, (r, c) in enumerate(free_states):
        if (r, c) == GOAL:
            continue
        best_val = -np.inf
        best_a   = 0
        for a in range(NUM_ACTIONS):
            q = expected_reward(r, c, a, deterministic)
            for prob, nr, nc in get_transitions(r, c, a, deterministic):
                q += GAMMA * prob * V[state_index[(nr, nc)]]
            if q > best_val:
                best_val = q
                best_a   = a
        policy[idx] = best_a
    return V, policy, conv_history


# ─────────────────────────────────────────────
# 7.  PLOTTING HELPERS
# ─────────────────────────────────────────────
def value_to_grid(V):
    """Map flat V vector back to a 2-D grid (NaN for walls)."""
    grid = np.full((ROWS, COLS), np.nan)
    for idx, (r, c) in enumerate(free_states):
        grid[r, c] = V[idx]
    return grid


def plot_policy(V, policy, deterministic, algo_name, ax):
    """Overlay arrow-field on the map (white background, black walls)."""
    # --- background: walls black, free cells white ---
    img = np.ones((ROWS, COLS, 3))          # white
    for r in range(ROWS):
        for c in range(COLS):
            if W[r, c] == 1:
                img[r, c] = [0, 0, 0]      # black wall

    ax.imshow(img, origin='upper', interpolation='nearest')

    # --- goal marker (bright green) ---
    gr, gc = GOAL
    ax.plot(gc, gr, 'o', color='#00e676', markersize=10, zorder=5)

    # --- arrow field ---
    scale  = 0.42   # arrow half-length in cell units
    for idx, (r, c) in enumerate(free_states):
        if (r, c) == GOAL:
            continue
        a  = policy[idx]
        dx = ARROW_DX[a] * scale
        dy = ARROW_DY[a] * scale
        ax.annotate('', xy=(c + dx, r + dy), xytext=(c - dx, r - dy),
                    arrowprops=dict(arrowstyle='->', color='#222222', lw=0.7))

    # --- grid lines for clarity ---
    for r in range(ROWS + 1):
        ax.axhline(r - 0.5, color='#cccccc', lw=0.25, zorder=0)
    for c in range(COLS + 1):
        ax.axvline(c - 0.5, color='#cccccc', lw=0.25, zorder=0)

    model_label = "Deterministic" if deterministic else "Stochastic"
    ax.set_title(f"{algo_name}  –  {model_label} Model", fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])


def plot_value_function(V, deterministic, algo_name, ax):
    """Heat-map of the value function (gray-scale like the reference)."""
    grid = value_to_grid(V)
    # mask walls
    masked = np.ma.array(grid, mask=np.isnan(grid))

    im = ax.imshow(masked, origin='upper', cmap='gray',
                   interpolation='nearest', aspect='equal')
    # goal marker
    gr, gc = GOAL
    ax.plot(gc, gr, 'o', color='#ff4444', markersize=8, zorder=5)

    # grid lines
    for r in range(ROWS + 1):
        ax.axhline(r - 0.5, color='#888888', lw=0.25, zorder=1)
    for c in range(COLS + 1):
        ax.axvline(c - 0.5, color='#888888', lw=0.25, zorder=1)

    model_label = "Deterministic" if deterministic else "Stochastic"
    ax.set_title(f"Value Function  –  {algo_name}  –  {model_label}", fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)


# ─────────────────────────────────────────────
# 8.  MAIN  – run everything and save figures
# ─────────────────────────────────────────────
if __name__ == '__main__':

    results = {}   # (algo, det) → (V, policy, conv)

    for det_flag, label in [(True, 'Deterministic'), (False, 'Stochastic')]:
        print(f"\n{'='*60}")
        print(f"  MODEL: {label}")
        print('='*60)

        # ── Policy Iteration ──
        t0 = time.time()
        V_pi, pol_pi, conv_pi = policy_iteration(det_flag)
        t_pi = time.time() - t0
        results[('PI', det_flag)] = (V_pi, pol_pi, conv_pi)
        print(f"\n[Policy Iteration – {label}]")
        print(f"  Policy-improvement rounds : {len(conv_pi)}")
        print(f"  Total eval sweeps         : {conv_pi[-1][0]}")
        print(f"  Wall-clock time           : {t_pi:.3f} s")

        # ── Value Iteration ──
        t0 = time.time()
        V_vi, pol_vi, conv_vi = value_iteration(det_flag)
        t_vi = time.time() - t0
        results[('VI', det_flag)] = (V_vi, pol_vi, conv_vi)
        print(f"\n[Value Iteration – {label}]")
        print(f"  Sweeps to convergence     : {len(conv_vi)}")
        print(f"  Final delta               : {conv_vi[-1][1]:.2e}")
        print(f"  Wall-clock time           : {t_vi:.3f} s")

    # ──────────────────────────────────────────
    # FIGURE 1: Policy maps (2 algos × 2 models = 4 panels)
    # ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(22, 7))
    fig.suptitle("Optimal Policy (Arrow Field)", fontsize=14, fontweight='bold', y=1.02)

    plot_policy(results[('PI', True)][0],  results[('PI', True)][1],  True,  "Policy Iteration", axes[0, 0])
    plot_policy(results[('VI', True)][0],  results[('VI', True)][1],  True,  "Value Iteration",  axes[0, 1])
    plot_policy(results[('PI', False)][0], results[('PI', False)][1], False, "Policy Iteration", axes[1, 0])
    plot_policy(results[('VI', False)][0], results[('VI', False)][1], False, "Value Iteration",  axes[1, 1])

    plt.tight_layout()
    plt.savefig('/home/claude/policy_maps.png', dpi=180, bbox_inches='tight')
    plt.close()
    print("\n  ✔  policy_maps.png saved")

    # ──────────────────────────────────────────
    # FIGURE 2: Value-function heat-maps (2 algos × 2 models)
    # ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(22, 7))
    fig.suptitle("Value Function (Gray-Scale Heat-Map)", fontsize=14, fontweight='bold', y=1.02)

    plot_value_function(results[('PI', True)][0],  True,  "Policy Iteration", axes[0, 0])
    plot_value_function(results[('VI', True)][0],  True,  "Value Iteration",  axes[0, 1])
    plot_value_function(results[('PI', False)][0], False, "Policy Iteration", axes[1, 0])
    plot_value_function(results[('VI', False)][0], False, "Value Iteration",  axes[1, 1])

    plt.tight_layout()
    plt.savefig('/home/claude/value_functions.png', dpi=180, bbox_inches='tight')
    plt.close()
    print("  ✔  value_functions.png saved")

    # ──────────────────────────────────────────
    # FIGURE 3: Convergence plots
    # ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.suptitle("Convergence Comparison", fontsize=13, fontweight='bold')

    # --- Value Iteration convergence (delta vs sweep) ---
    ax = axes[0]
    for det_flag, color, ls in [(True, '#1976d2', '-'), (False, '#e53935', '--')]:
        conv = results[('VI', det_flag)][2]
        sweeps = [x[0] for x in conv]
        deltas = [x[1] for x in conv]
        lbl = "Deterministic" if det_flag else "Stochastic"
        ax.semilogy(sweeps, deltas, color=color, linestyle=ls, linewidth=2, label=f"VI – {lbl}")
    ax.set_xlabel("Sweep (iteration)", fontsize=11)
    ax.set_ylabel("Max Δ (log scale)", fontsize=11)
    ax.set_title("Value Iteration – Convergence", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', ls=':', alpha=0.5)

    # --- Policy Iteration: eval-sweeps per improvement round ---
    ax = axes[1]
    for det_flag, color, ls in [(True, '#1976d2', '-'), (False, '#e53935', '--')]:
        conv = results[('PI', det_flag)][2]
        rounds       = list(range(1, len(conv) + 1))
        cum_sweeps   = [x[0] for x in conv]
        lbl = "Deterministic" if det_flag else "Stochastic"
        ax.plot(rounds, cum_sweeps, 'o-', color=color, linestyle=ls, linewidth=2,
                markersize=5, label=f"PI – {lbl}")
    ax.set_xlabel("Policy-improvement round", fontsize=11)
    ax.set_ylabel("Cumulative eval sweeps", fontsize=11)
    ax.set_title("Policy Iteration – Cumulative Eval Sweeps", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, ls=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig('/home/claude/convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✔  convergence.png saved")

    print("\n  All figures generated successfully.")
