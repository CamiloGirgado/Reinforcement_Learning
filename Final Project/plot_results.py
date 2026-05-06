import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy

def moving_average(values, window):
    """Smooths the data for better visualization in slides."""
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_comparison(log_dirs, labels, title="DRL Performance Comparison"):
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e'] # Blue for Baseline, Orange for Recurrent
    
    for log_dir, label, color in zip(log_dirs, labels, colors):
        # 1. Load data from the Monitor CSV files
        df = load_results(log_dir)
        x, y = ts2xy(df, 'timesteps')
        
        # 2. Smooth the rewards (window size of 50 episodes)
        y_smoothed = moving_average(y, window=50)
        x_smoothed = x[len(x) - len(y_smoothed):] # Align x-axis
        
        # 3. Plot
        plt.plot(x_smoothed, y_smoothed, label=label, color=color, linewidth=2)
        plt.fill_between(x_smoothed, y_smoothed - np.std(y_smoothed)*0.1, 
                         y_smoothed + np.std(y_smoothed)*0.1, color=color, alpha=0.1)

    plt.xlabel('Total Timesteps')
    plt.ylabel('Smoothed Episodic Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("./plots/final_comparison.png")
    plt.show()

if __name__ == "__main__":
    # Point these to the directories where your Monitor files are saved
    log_paths = ["./models/ppo_logs/", "./models/recurrent_ppo_logs/"]
    names = ["Baseline PPO (Frame Stacking)", "Recurrent PPO (LSTM Memory)"]
    
    import os
    os.makedirs("./plots/", exist_ok=True)
    plot_comparison(log_paths, names)