import os
import gymnasium as gym
import numpy as np
import imageio
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from train_recurrent_ppo import make_single_env

def record_recurrent_simulation(model_path, output_path="./videos/recurrent_agent.mp4", num_episodes=3):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Recreate the non-stacked evaluation environment
    env_fn = make_single_env(use_grayscale=False)
    eval_env = DummyVecEnv([env_fn])
    
    # Load the trained Recurrent PPO model
    model = RecurrentPPO.load(model_path, env=eval_env)
    
    frames = []
    obs = eval_env.reset()
    
    # LSTM states must start as None (zero-initialized internally)
    lstm_states = None
    
    # Keep track of episode starts to reset LSTM states when an agent dies
    num_envs = eval_env.num_envs
    episode_starts = np.ones((num_envs,), dtype=bool)
    
    print("Recording evaluation run for Recurrent Agent...")
    episodes_recorded = 0
    
    while episodes_recorded < num_episodes:
        # Crucial: Pass the current hidden states and episode flags to the model
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states, 
            episode_start=episode_starts, 
            deterministic=True
        )
        
        obs, rewards, dones, infos = eval_env.step(action)
        
        # Reset tracker flag (if done, the next step is the start of a new episode)
        episode_starts = dones
        
        # Grab raw frame for video
        frame = eval_env.envs[0].unwrapped.render()
        frames.append(frame)
        
        if dones[0]:
            episodes_recorded += 1
            print(f"Episode {episodes_recorded} Complete.")
            
    print(f"Saving recurrent simulation video to {output_path}...")
    imageio.mimsave(output_path, [np.array(f) for f in frames], fps=24)
    print("Recurrent Agent Video Saved Successfully!")

if __name__ == "__main__":
    record_recurrent_simulation(model_path="./models/recurrent_ppo_best/recurrent_ppo_final")