"""
DQN Agent for LunarLander-v2
Programming Exercise #4 - Deep Q-Network

Improvements over starter code:
  - Step-based target network updates (more stable than episode-based)
  - TensorBoard logging for reward, epsilon, and loss
  - iter_cntr used to gate target network sync
  - Evaluation/test mode separated from training
  - Compatible with both old gym and newer gymnasium APIs
"""

import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import os
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------------------
# Epsilon-greedy schedule (linear decay)
# ---------------------------------------------------------------------------

class EpsilonDecayScheduler:
    """Linearly decays epsilon from eps_start down to eps_finish
    over eps_aneal_time environment steps."""

    def __init__(self, eps_start: float, eps_finish: float, eps_aneal_time: int):
        self.eps_start = eps_start
        self.eps_finish = eps_finish
        self.eps_aneal_time = eps_aneal_time
        self.dd = (eps_start - eps_finish) / eps_aneal_time

    def get_epsilon(self, current_time_step: int) -> float:
        return max(self.eps_finish, self.eps_start - self.dd * current_time_step)


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class DeepQNetwork(nn.Module):
    """Two-hidden-layer MLP that maps states → Q-values for every action."""

    def __init__(self, input_dims, fc1_dims: int, fc2_dims: int, n_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.fc1.weight.dtype)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(
        self,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_aneal_time: int,
        lr: float,
        input_dims,
        batch_size: int,
        n_actions: int,
        replay_buffer_size: int,
        fc1_dims: int = 256,
        fc2_dims: int = 256,
        target_update_steps: int = 1000,   # hard-copy target every N env steps
    ):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.batch_size = batch_size
        self.target_update_steps = target_update_steps

        self.action_space = list(range(n_actions))
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer_counter = 0
        self.iter_cntr = 0          # counts learn() calls, used for target sync

        # Networks
        self.Q_eval = DeepQNetwork(input_dims, fc1_dims, fc2_dims, n_actions)
        self.Q_target = deepcopy(self.Q_eval)
        self.Q_target.eval()        # target never trained directly

        self.optimizer = optim.Adam(self.Q_eval.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.epsilon_scheduler = EpsilonDecayScheduler(
            epsilon_start, epsilon_end, epsilon_aneal_time
        )

        # Replay buffer (pre-allocated numpy arrays for speed)
        self.state_memory     = np.zeros((replay_buffer_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((replay_buffer_size, *input_dims), dtype=np.float32)
        self.action_memory    = np.zeros(replay_buffer_size, dtype=np.int32)
        self.reward_memory    = np.zeros(replay_buffer_size, dtype=np.float32)
        self.terminal_memory  = np.zeros(replay_buffer_size, dtype=bool)

        total_params = sum(p.numel() for p in self.Q_eval.parameters())
        print(f"[DQN] Total network parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # Replay buffer
    # ------------------------------------------------------------------

    def store_transition(self, state, action, reward, state_, terminal):
        idx = self.replay_buffer_counter % self.replay_buffer_size
        self.state_memory[idx]     = state
        self.new_state_memory[idx] = state_
        self.action_memory[idx]    = action
        self.reward_memory[idx]    = reward
        self.terminal_memory[idx]  = terminal
        self.replay_buffer_counter += 1

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, observation, curr_env_step: int = 0, explore: bool = True):
        if explore:
            self.epsilon = self.epsilon_scheduler.get_epsilon(curr_env_step)
            if np.random.random() > self.epsilon:
                state = torch.tensor([observation]).to(self.Q_eval.device)
                with torch.no_grad():
                    action = torch.argmax(self.Q_eval(state)).item()
            else:
                action = np.random.choice(self.action_space)
        else:
            state = torch.tensor([observation]).to(self.Q_eval.device)
            with torch.no_grad():
                action = torch.argmax(self.Q_eval(state)).item()
        return action

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------

    def learn(self) -> float | None:
        """Performs one gradient update. Returns scalar loss, or None if buffer
        is not yet large enough."""
        if self.replay_buffer_counter < self.batch_size:
            return None

        self.optimizer.zero_grad()

        max_mem = min(self.replay_buffer_counter, self.replay_buffer_size)
        batch   = np.random.choice(max_mem, self.batch_size, replace=False)

        states      = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_states  = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        rewards     = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminals   = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        actions     = self.action_memory[batch]
        batch_idx   = np.arange(self.batch_size, dtype=np.int32)

        # Q(s, a) from online network
        q_eval = self.Q_eval(states)[batch_idx, actions]

        # max Q(s', ·) from frozen target network
        with torch.no_grad():
            q_next = self.Q_target(new_states)
            q_next[terminals] = 0.0
            q_target = rewards + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.loss_fn(q_target, q_eval)
        loss.backward()
        self.optimizer.step()

        self.iter_cntr += 1

        # Hard-copy target network every `target_update_steps` learn() calls
        if self.iter_cntr % self.target_update_steps == 0:
            self.Q_target.load_state_dict(deepcopy(self.Q_eval.state_dict()))
            print(f"  [DQN] Target network synced at learn step {self.iter_cntr}.")

        return loss.item()

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def save_models(self, path: str = "."):
        torch.save(self.Q_eval.state_dict(),   f"{path}/dqn_q_eval.pth")
        torch.save(self.Q_target.state_dict(), f"{path}/dqn_q_target.pth")

    def load_models(self, path: str = "."):
        self.Q_eval.load_state_dict(torch.load(f"{path}/dqn_q_eval.pth"))
        self.Q_target.load_state_dict(torch.load(f"{path}/dqn_q_target.pth"))


# ---------------------------------------------------------------------------
# Helper: handle both old gym and newer gymnasium reset/step signatures
# ---------------------------------------------------------------------------

def env_reset(env):
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]          # gymnasium: (obs, info)
    return result                 # gym: obs

def env_step(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result   # gymnasium
        return obs, reward, terminated or truncated, info
    return result                                           # gym: obs, r, done, info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # Hyperparameters
    # Tuning notes:
    #   - replay_buffer_size=100 000: balances memory and diversity
    #   - epsilon_aneal_time=200 000 steps: long enough to explore well
    #   - target_update_steps=1 000: stable target every ~16 episodes early on
    #   - lr=5e-4: slightly lower than default Adam lr for smoother convergence
    #   - batch_size=64: standard; increase to 128 if training is noisy
    # -----------------------------------------------------------------------
    GAMMA              = 0.99
    LR                 = 5e-4
    BATCH_SIZE         = 64
    REPLAY_BUFFER_SIZE = 100_000
    EPSILON_START      = 1.0
    EPSILON_END        = 0.01
    EPSILON_ANEAL_TIME = 200_000   # env steps over which epsilon decays
    TARGET_UPDATE_STEPS = 1_000    # learn() calls between hard target syncs
    N_EPISODES         = 2_000
    FC1_DIMS           = 256
    FC2_DIMS           = 256
    SAVE_INTERVAL      = 100       # episodes between model checkpoints
    # -----------------------------------------------------------------------

    env = gym.make("LunarLander-v2")
    input_dims = [env.observation_space.shape[0]]   # [8]
    n_actions  = env.action_space.n                  # 4

    agent = Agent(
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_aneal_time=EPSILON_ANEAL_TIME,
        lr=LR,
        input_dims=input_dims,
        batch_size=BATCH_SIZE,
        n_actions=n_actions,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        fc1_dims=FC1_DIMS,
        fc2_dims=FC2_DIMS,
        target_update_steps=TARGET_UPDATE_STEPS,
    )

    # TensorBoard writer — auto-increment run folder
    logs_root = Path(os.path.dirname(os.path.abspath(__file__))) / "logs_dqn"
    if not logs_root.exists():
        run_name = "run001"
    else:
        existing = [
            int(f.name.replace("run", ""))
            for f in logs_root.iterdir()
            if f.name.startswith("run") and f.name[3:].isdigit()
        ]
        run_name = f"run{(max(existing) + 1 if existing else 1):03d}"
    logs_dir = logs_root / run_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(logs_dir))
    print(f"[DQN] TensorBoard logs → {logs_dir}")

    total_env_steps = 0
    score_history   = []

    for episode in range(N_EPISODES):
        obs   = env_reset(env)
        score = 0.0
        done  = False
        ep_losses = []

        while not done:
            action                      = agent.choose_action(obs, total_env_steps, explore=True)
            obs_, reward, done, info    = env_step(env, action)
            agent.store_transition(obs, action, reward, obs_, done)
            loss = agent.learn()
            if loss is not None:
                ep_losses.append(loss)

            obs             = obs_
            score          += reward
            total_env_steps += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        # TensorBoard
        writer.add_scalar("Reward/episode",         score,              total_env_steps)
        writer.add_scalar("Reward/avg100",          avg_score,          total_env_steps)
        writer.add_scalar("Exploration/epsilon",    agent.epsilon,      total_env_steps)
        if ep_losses:
            writer.add_scalar("Loss/mean_per_episode", np.mean(ep_losses), total_env_steps)

        print(
            f"Episode {episode:4d} | steps {total_env_steps:7d} | "
            f"score {score:8.1f} | avg100 {avg_score:8.1f} | ε {agent.epsilon:.4f}"
        )

        if (episode + 1) % SAVE_INTERVAL == 0:
            agent.save_models(str(logs_dir))
            print(f"  [DQN] Models saved at episode {episode + 1}.")

    writer.close()
    env.close()
    print("[DQN] Training complete.")
