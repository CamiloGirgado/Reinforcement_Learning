import os
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# Create directory structures
os.makedirs("./models/recurrent_ppo_logs/", exist_ok=True)
os.makedirs("./models/recurrent_ppo_checkpoints/", exist_ok=True)
os.makedirs("./models/recurrent_ppo_best/", exist_ok=True)

def make_single_env(env_id="MiniWorld-CollectHealth-v0", use_grayscale=False):
    """
    Creates a single environment wrapper. 
    Notice: NO FrameStacking wrapper here, the LSTM handles memory!
    """
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)
        env = ResizeObservation(env, shape=(84, 84))
        if use_grayscale:
            env = GrayScaleObservation(env, keep_dim=True)
        return env
    return _init

def main():
    # 1. Initialize parallel environments
    num_envs = 4
    env_fns = [make_single_env(use_grayscale=False) for _ in range(num_envs)]
    vec_env = DummyVecEnv(env_fns)

    # 2. Configure Recurrent PPO Hyperparameters
    recurrent_ppo_hyperparams = {
        "policy": "CnnLstmPolicy",          # Uses a CNN feature extractor followed by an LSTM layer
        "env": vec_env,
        "learning_rate": 2.5e-4,
        "n_steps": 128,                     # Rollout steps collected per env per update
        "batch_size": 128,                  # Update batch size
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.03,                   # Slightly higher exploration for search behavior
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 1,
        "tensorboard_log": "./models/recurrent_ppo_logs/"
    }

    # Initialize RecurrentPPO model
    model = RecurrentPPO(**recurrent_ppo_hyperparams)

    # 3. Setup Checkpoint Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // num_envs, 1),
        save_path="./models/recurrent_ppo_checkpoints/",
        name_prefix="recurrent_ppo_miniworld"
    )

    # 4. Start Training
    total_timesteps = 500_000
    print(f"Starting RECURRENT PPO training for {total_timesteps} steps...")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="Recurrent_PPO_CollectHealth"
    )

    # Save final recurrent model
    model.save("./models/recurrent_ppo_best/recurrent_ppo_final")
    print("Recurrent PPO Training Complete. Model Saved!")

if __name__ == "__main__":
    main()