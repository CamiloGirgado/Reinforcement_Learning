import numpy as np
import matplotlib.pyplot as plt

# --- Environment Configuration ---
# 1 = occupied (obstacle), 0 = free 
W = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,0,1,1,1,1,1,1,1],
    [1,0,0,0,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
]) 

GOAL = (8, 11) # Red 0 is at row 8, col 11 
GAMMA = 0.95 
REWARD_GOAL = 100.0 
REWARD_OBSTACLE = -50.0 
REWARD_STEP = -1.0 

# 8 directions: N, NE, E, SE, S, SW, W, NW 
ACTIONS = [(-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)]

def get_transitions(s, a_idx, stochastic):
    if s == GOAL: return [(1.0, s, 0)]
    
    if not stochastic: # Deterministic 
        probs = [(1.0, a_idx)]
    else: # Stochastic (20% chance of +/- 45 deg) 
        probs = [(0.8, a_idx), (0.1, (a_idx-1)%8), (0.1, (a_idx+1)%8)]
    
    results = []
    for prob, idx in probs:
        move = ACTIONS[idx]
        next_s = (s[0] + move[0], s[1] + move[1])
        if (next_s[0] < 0 or next_s[0] >= W.shape[0] or 
            next_s[1] < 0 or next_s[1] >= W.shape[1] or W[next_s] == 1):
            results.append((prob, next_s, REWARD_OBSTACLE))
        elif next_s == GOAL:
            results.append((prob, next_s, REWARD_GOAL))
        else:
            results.append((prob, next_s, REWARD_STEP))
    return results

def value_iteration(stochastic=False):
    V = np.zeros(W.shape)
    iters = 0
    while True:
        delta = 0
        new_V = np.copy(V)
        for r in range(W.shape[0]):
            for c in range(W.shape[1]):
                if (r,c) == GOAL or W[r,c] == 1: continue
                v_actions = [sum(p * (rew + GAMMA * V[ns]) for p, ns, rew in get_transitions((r,c), ai, stochastic)) 
                             for ai in range(8)]
                best_v = max(v_actions)
                delta = max(delta, abs(best_v - V[r,c]))
                new_V[r,c] = best_v
        V = new_V
        iters += 1
        if delta < 1e-4: break
    
    policy = np.zeros(W.shape, dtype=int)
    for r in range(W.shape[0]):
        for c in range(W.shape[1]):
            if W[r,c] == 1 or (r,c) == GOAL: continue
            v_actions = [sum(p * (rew + GAMMA * V[ns]) for p, ns, rew in get_transitions((r,c), ai, stochastic)) 
                         for ai in range(8)]
            policy[r,c] = np.argmax(v_actions)
    return V, policy, iters

# --- Plotting Functions ---

def plot_value_function_only(V, title):
    """Standalone plot of the Value Function"""
    plt.figure(figsize=(12, 4))
    V_masked = np.ma.masked_where(W == 1, V)
    plt.imshow(V_masked, cmap='bone') 
    plt.title(f"Value Function: {title}")
    plt.colorbar(label='Value')
    plt.show()

def plot_optimal_policy(V, policy, title):
    """Plot of the Optimal Policy"""
    plt.figure(figsize=(12, 4))
    plt.imshow(W, cmap='gray_r')
    for r in range(W.shape[0]):
        for c in range(W.shape[1]):
            if W[r,c] == 0 and (r,c) != GOAL:
                dr, dc = ACTIONS[policy[r,c]]
                plt.arrow(c, r, dc*0.3, dr*0.3, head_width=0.2, color='blue')
    plt.plot(GOAL[1], GOAL[0], 'ro', markersize=8)
    plt.title(f"Optimal Policy: {title}")
    plt.show()

# --- Main Execution ---
# Deterministic Model
V_det, P_det, it_det = value_iteration(stochastic=False)
print(f"Deterministic converged in {it_det} iterations.")
plot_value_function_only(V_det, "Deterministic Model")  
plot_optimal_policy(V_det, P_det, "Deterministic Model (a)")

# Stochastic Model
V_sto, P_sto, it_sto = value_iteration(stochastic=True)
print(f"Stochastic converged in {it_sto} iterations.")
plot_value_function_only(V_sto, "Stochastic Model")  
plot_optimal_policy(V_sto, P_sto, "Stochastic Model (b)")