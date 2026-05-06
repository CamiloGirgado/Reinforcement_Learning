import os
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Create directory structures for organization
os.makedirs("./models/ppo_logs/", exist_ok=True)
os.makedirs("./models/ppo_checkpoints/", exist_ok=True)
os.makedirs("./models/ppo_best/", exist_ok=True)

def make_single_env(env_id="MiniWorld-CollectHealth-v0", use_grayscale=False):
    """
    Helper function to instantiate and preprocess a single MiniWorld environment.
    """
    def _init():
        # 1. Instantiate the raw 3D environment
        env = gym.make(env_id)
        
        # 2. Monitor wrapper tracks training episode lengths, rewards, and times
        env = Monitor(env)
        
        # 3. Resize from 60x80 to 84x84 (Atari standard size for CNN feature extraction)
        env = ResizeObservation(env, shape=(84, 84))
        
        # 4. Optional: Convert to grayscale to speed up training
        if use_grayscale:
            env = GrayScaleObservation(env, keep_dim=True)
            
        return env
    return _init

def main():
    # --- 1. Environment Vectorization & Frame Stacking ---
    # We use 4 environments running in parallel to speed up rollout collection.
    # If running headless or running into OpenGL errors, DummyVecEnv is highly stable.
    num_envs = 4
    env_fns = [make_single_env(use_grayscale=False) for _ in range(num_envs)]
    vec_env = DummyVecEnv(env_fns)
    
    # Apply Frame Stacking at the vectorized environment level.
    # Stacking 4 frames gives the CNN a history/trajectory of what is happening.
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order="last")

    # --- 2. Define PPO Hyperparameters ---
    # These parameters are tuned specifically for finding health packs in 3D:
    ppo_hyperparams = {
        "policy": "CnnPolicy",             # Standard NatureCNN for pixel/image inputs
        "env": vec_env,
        "learning_rate": 2.5e-4,           # Standard start for visual tasks
        "n_steps": 1024,                   # Steps per env before running an update
        "batch_size": 128,                 # Minibatch size for gradient updates
        "n_epochs": 10,                    # Number of optimization epochs
        "gamma": 0.99,                     # Moderately long discount factor
        "gae_lambda": 0.95,                # GAE smoothing factor
        "clip_range": 0.2,                 # PPO policy clipping threshold
        "ent_coef": 0.02,                  # Crucial! Non-zero entropy forces exploration 
        "vf_coef": 0.5,                    # Value function coefficient
        "max_grad_norm": 0.5,              # Gradient clipping max
        "verbose": 1,
        "tensorboard_log": "./models/ppo_logs/"
    }

    # Initialize the PPO agent
    model = PPO(**ppo_hyperparams)

    # --- 3. Set Up Callbacks ---
    # Automatically saves checkpoints every 50,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // num_envs, 1),
        save_path="./models/ppo_checkpoints/",
        name_prefix="ppo_miniworld"
    )

    # --- 4. Start Training ---
    total_timesteps = 500_000  # Expand to 1M or 2M for final presentation metrics
    print(f"Starting training for {total_timesteps} total steps across {num_envs} environments...")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="PPO_CollectHealth_Baseline"
    )

    # Save final model
    model.save("./models/ppo_best/ppo_collect_health_final")
    print("Training Complete. Model Saved!")

if __name__ == "__main__":
    main()