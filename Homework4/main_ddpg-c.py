"""
DDPG Agent for LunarLanderContinuous-v2
Programming Exercise #4 - Deep Deterministic Policy Gradient

Improvements over starter code:
  - Ornstein-Uhlenbeck (OU) exploration noise with sigma decay
    (replaces simple torch.rand noise; OU noise is temporally correlated,
     which suits continuous control tasks)
  - Target networks hard-copied (τ=1) at init, then soft-updated each step
  - Batch normalisation in actor and critic for more stable training
  - TensorBoard logging: reward, critic loss, actor loss, noise sigma
  - Compatible with both old gym and newer gymnasium APIs
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import os


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck noise with sigma decay
# ---------------------------------------------------------------------------

class OUNoise:
    """Temporally correlated noise for continuous action-space exploration.

    Parameters
    ----------
    n_actions : number of action dimensions
    mu        : long-run mean of the noise process (usually 0)
    theta     : rate of mean reversion (higher → reverts faster)
    sigma     : initial noise magnitude
    sigma_min : floor for sigma after decay
    decay     : multiplicative decay applied each episode reset
    dt        : time-step size
    """

    def __init__(
        self,
        n_actions: int,
        mu: float       = 0.0,
        theta: float    = 0.15,
        sigma: float    = 0.3,
        sigma_min: float = 0.05,
        decay: float    = 0.995,
        dt: float       = 1e-2,
    ):
        self.mu        = mu * np.ones(n_actions)
        self.theta     = theta
        self.sigma     = sigma
        self.sigma_min = sigma_min
        self.decay     = decay
        self.dt        = dt
        self.state     = self.mu.copy()

    def reset(self):
        """Reset noise state and apply sigma decay (call once per episode)."""
        self.state = self.mu.copy()
        self.sigma = max(self.sigma_min, self.sigma * self.decay)

    def sample(self) -> np.ndarray:
        dx = (
            self.theta * (self.mu - self.state) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.randn(*self.state.shape)
        )
        self.state = self.state + dx
        return self.state.copy()


# ---------------------------------------------------------------------------
# Actor network
# ---------------------------------------------------------------------------

class ActorNetwork(nn.Module):
    """
    Maps state → deterministic action ∈ [-1, 1]^n_actions  (via tanh).
    Batch normalisation after each hidden layer improves training stability.
    """

    def __init__(self, alpha: float, state_dim: int, fc1_dim: int, fc2_dim: int, n_actions: int):
        super().__init__()
        self.fc1  = nn.Linear(state_dim, fc1_dim)
        self.bn1  = nn.LayerNorm(fc1_dim)
        self.fc2  = nn.Linear(fc1_dim, fc2_dim)
        self.bn2  = nn.LayerNorm(fc2_dim)
        self.mu   = nn.Linear(fc2_dim, n_actions)

        # Initialise final layer with small weights so initial actions are ~0
        nn.init.uniform_(self.mu.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mu.bias,   -3e-3, 3e-3)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x  = F.relu(self.bn1(self.fc1(state)))
        x  = F.relu(self.bn2(self.fc2(x)))
        return torch.tanh(self.mu(x))


# ---------------------------------------------------------------------------
# Critic network
# ---------------------------------------------------------------------------

class CriticNetwork(nn.Module):
    """
    Maps (state, action) → scalar Q-value.
    Actions are concatenated after the first hidden layer (common practice).
    """

    def __init__(self, beta: float, state_dim: int, fc1_dim: int, fc2_dim: int, n_actions: int):
        super().__init__()
        self.fc1  = nn.Linear(state_dim, fc1_dim)
        self.bn1  = nn.LayerNorm(fc1_dim)
        # action injected here
        self.fc2  = nn.Linear(fc1_dim + n_actions, fc2_dim)
        self.bn2  = nn.LayerNorm(fc2_dim)
        self.q    = nn.Linear(fc2_dim, 1)

        nn.init.uniform_(self.q.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q.bias,   -3e-3, 3e-3)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(torch.cat([x, action], dim=1))))
        return self.q(x)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, max_size: int, state_dim: int, n_actions: int):
        self.mem_size      = max_size
        self.mem_cntr      = 0
        self.state_memory  = np.zeros((max_size, state_dim), dtype=np.float32)
        self.next_state_memory = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action_memory = np.zeros((max_size, n_actions),  dtype=np.float32)
        self.reward_memory = np.zeros(max_size,               dtype=np.float32)
        self.terminal_memory = np.zeros(max_size,             dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        idx = self.mem_cntr % self.mem_size
        self.state_memory[idx]      = state
        self.next_state_memory[idx] = state_
        self.action_memory[idx]     = action
        self.reward_memory[idx]     = reward
        self.terminal_memory[idx]   = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size: int):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch   = np.random.choice(max_mem, batch_size, replace=False)
        return (
            self.state_memory[batch],
            self.action_memory[batch],
            self.reward_memory[batch],
            self.next_state_memory[batch],
            self.terminal_memory[batch],
        )


# ---------------------------------------------------------------------------
# DDPG Agent
# ---------------------------------------------------------------------------

class DDPGAgent:
    def __init__(
        self,
        state_dim:  int,
        n_actions:  int,
        alpha:      float = 1e-4,     # actor  lr  (lower than critic)
        beta:       float = 1e-3,     # critic lr
        gamma:      float = 0.99,
        tau:        float = 0.005,    # soft-update coefficient
        fc1_dim:    int   = 400,
        fc2_dim:    int   = 300,
    ):
        self.gamma    = gamma
        self.tau      = tau
        self.n_actions = n_actions

        self.actor        = ActorNetwork(alpha, state_dim, fc1_dim, fc2_dim, n_actions)
        self.critic       = CriticNetwork(beta,  state_dim, fc1_dim, fc2_dim, n_actions)
        self.target_actor  = ActorNetwork(alpha, state_dim, fc1_dim, fc2_dim, n_actions)
        self.target_critic = CriticNetwork(beta,  state_dim, fc1_dim, fc2_dim, n_actions)

        # Hard copy online → target at initialisation
        self._hard_update(self.target_actor,  self.actor)
        self._hard_update(self.target_critic, self.critic)

        self.noise = OUNoise(
            n_actions=n_actions,
            mu=0.0,
            theta=0.15,
            sigma=0.3,
            sigma_min=0.05,
            decay=0.995,
        )

        # Low/high bounds for LunarLanderContinuous-v2
        self.action_low  = np.array([0.0, -1.0])   # main engine [0,1], side [-1,1]
        self.action_high = np.array([1.0,  1.0])

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _hard_update(target: nn.Module, source: nn.Module):
        target.load_state_dict(source.state_dict())

    def _soft_update(self):
        for tp, sp in zip(self.target_actor.parameters(), self.actor.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
        for tp, sp in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, observation, explore: bool = True) -> np.ndarray:
        self.actor.eval()  # disable BatchNorm running-stat updates during inference
        state   = torch.tensor(observation, dtype=torch.float).unsqueeze(0).to(self.actor.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        self.actor.train()

        if explore:
            action = action + self.noise.sample()

        return np.clip(action, self.action_low, self.action_high)

    def reset_noise(self):
        """Call once per episode to reset & decay OU noise."""
        self.noise.reset()

    # ------------------------------------------------------------------
    # Learning step — returns (critic_loss, actor_loss)
    # ------------------------------------------------------------------

    def learn(self, memory: ReplayBuffer, batch_size: int):
        if memory.mem_cntr < batch_size:
            return None, None

        states, actions, rewards, next_states, dones = memory.sample_buffer(batch_size)

        dev = self.actor.device
        states      = torch.tensor(states,      dtype=torch.float).to(dev)
        actions     = torch.tensor(actions,     dtype=torch.float).to(dev)
        rewards     = torch.tensor(rewards,     dtype=torch.float).to(dev)
        next_states = torch.tensor(next_states, dtype=torch.float).to(dev)
        dones       = torch.tensor(dones,       dtype=torch.bool ).to(dev)

        # ---- Critic update ------------------------------------------------
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.train()

        with torch.no_grad():
            next_actions  = self.target_actor(next_states)
            q_next        = self.target_critic(next_states, next_actions).squeeze()
            q_next[dones] = 0.0
            q_target      = rewards + self.gamma * q_next

        q_eval       = self.critic(states, actions).squeeze()
        critic_loss  = F.mse_loss(q_eval, q_target)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # ---- Actor update -------------------------------------------------
        self.critic.eval()
        self.actor.train()

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # ---- Soft update targets ------------------------------------------
        self._soft_update()

        return critic_loss.item(), actor_loss.item()

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def save_models(self, path: str = "."):
        torch.save(self.actor.state_dict(),        f"{path}/ddpg_actor.pth")
        torch.save(self.critic.state_dict(),       f"{path}/ddpg_critic.pth")
        torch.save(self.target_actor.state_dict(), f"{path}/ddpg_target_actor.pth")
        torch.save(self.target_critic.state_dict(),f"{path}/ddpg_target_critic.pth")

    def load_models(self, path: str = "."):
        self.actor.load_state_dict(       torch.load(f"{path}/ddpg_actor.pth"))
        self.critic.load_state_dict(      torch.load(f"{path}/ddpg_critic.pth"))
        self.target_actor.load_state_dict(torch.load(f"{path}/ddpg_target_actor.pth"))
        self.target_critic.load_state_dict(torch.load(f"{path}/ddpg_target_critic.pth"))


# ---------------------------------------------------------------------------
# Helper: handle both old gym and newer gymnasium reset/step signatures
# ---------------------------------------------------------------------------

def env_reset(env):
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result

def env_step(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, terminated or truncated, info
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # Hyperparameters
    # Tuning notes:
    #   - alpha=1e-4 (actor), beta=1e-3 (critic): actor moves more slowly so
    #     the critic Q-estimate stabilises before the policy shifts too much
    #   - tau=0.005: small value → slow, smooth target tracking
    #   - sigma=0.3 → 0.05 over episodes: high initial exploration, then refine
    #   - batch_size=128: larger batches reduce gradient variance for DDPG
    #   - buffer_size=1 000 000: large enough to avoid forgetting early experience
    # -----------------------------------------------------------------------
    ALPHA          = 1e-4        # actor learning rate
    BETA           = 1e-3        # critic learning rate
    GAMMA          = 0.99
    TAU            = 0.005
    BATCH_SIZE     = 128
    BUFFER_SIZE    = 1_000_000
    FC1_DIM        = 400
    FC2_DIM        = 300
    N_EPISODES     = 1_500
    MAX_STEPS      = 1_000       # max steps per episode
    SAVE_INTERVAL  = 100
    # -----------------------------------------------------------------------

    env       = gym.make("LunarLanderContinuous-v2")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    agent  = DDPGAgent(state_dim, n_actions, alpha=ALPHA, beta=BETA,
                       gamma=GAMMA, tau=TAU, fc1_dim=FC1_DIM, fc2_dim=FC2_DIM)
    memory = ReplayBuffer(BUFFER_SIZE, state_dim, n_actions)

    # TensorBoard writer
    logs_root = Path(os.path.dirname(os.path.abspath(__file__))) / "logs_ddpg"
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
    print(f"[DDPG] TensorBoard logs → {logs_dir}")

    score_history   = []
    total_env_steps = 0

    for episode in range(N_EPISODES):
        obs   = env_reset(env)
        agent.reset_noise()           # decay & reset OU noise each episode
        score = 0.0
        done  = False
        step  = 0
        ep_critic_losses = []
        ep_actor_losses  = []

        while not done and step < MAX_STEPS:
            action              = agent.choose_action(obs, explore=True)
            obs_, reward, done, info = env_step(env, action)
            memory.store_transition(obs, action, reward, obs_, done)

            c_loss, a_loss = agent.learn(memory, BATCH_SIZE)
            if c_loss is not None:
                ep_critic_losses.append(c_loss)
                ep_actor_losses.append(a_loss)

            obs             = obs_
            score          += reward
            total_env_steps += 1
            step           += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        # TensorBoard
        writer.add_scalar("Reward/episode",          score,                  total_env_steps)
        writer.add_scalar("Reward/avg100",           avg_score,              total_env_steps)
        writer.add_scalar("Noise/sigma",             agent.noise.sigma,      total_env_steps)
        if ep_critic_losses:
            writer.add_scalar("Loss/critic", np.mean(ep_critic_losses), total_env_steps)
            writer.add_scalar("Loss/actor",  np.mean(ep_actor_losses),  total_env_steps)

        print(
            f"Episode {episode:4d} | steps {total_env_steps:7d} | "
            f"score {score:8.1f} | avg100 {avg_score:8.1f} | σ {agent.noise.sigma:.4f}"
        )

        if (episode + 1) % SAVE_INTERVAL == 0:
            agent.save_models(str(logs_dir))
            print(f"  [DDPG] Models saved at episode {episode + 1}.")

    writer.close()
    env.close()
    print("[DDPG] Training complete.")
