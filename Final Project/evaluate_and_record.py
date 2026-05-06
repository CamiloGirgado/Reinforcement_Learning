import os
import gymnasium as gym
import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from train_sb3_ppo import make_single_env  # Import our wrapper helper

def record_simulation(model_path, output_path="./videos/trained_agent.mp4", num_episodes=3):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Recreate the exact environment pipeline used during training
    env_fn = make_single_env(use_grayscale=False)
    eval_env = DummyVecEnv([env_fn])
    eval_env = VecFrameStack(eval_env, n_stack=4, channels_order="last")
    
    # Load the trained model
    model = PPO.load(model_path, env=eval_env)
    
    frames = []
    obs = eval_env.reset()
    
    print("Recording evaluation run...")
    episodes_recorded = 0
    
    while episodes_recorded < num_episodes:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)
        
        # Render the raw pixel observation from the base env (not the stacked, preprocessed one)
        # Note: Accessing the base environment to get high-quality render
        frame = eval_env.envs[0].unwrapped.render()
        frames.append(frame)
        
        if dones[0]:
            episodes_recorded += 1
            print(f"Episode {episodes_recorded} Complete.")
            
    # Save compilation to a high-quality MP4 video
    print(f"Saving simulation video to {output_path}...")
    imageio.mimsave(output_path, [np.array(f) for f in frames], fps=24)
    print("Video Saved Successfully!")

if __name__ == "__main__":
    # Test with the final saved model
    record_simulation(model_path="./models/ppo_best/ppo_collect_health_final")