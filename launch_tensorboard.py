#!/usr/bin/env python
"""Launch TensorBoard to visualize training logs."""

import subprocess
import os
import sys

# Set the logs directory
logs_dir = os.path.join(os.path.dirname(__file__), 'logs')

print(f"Launching TensorBoard with logs from: {logs_dir}")
print("TensorBoard will be available at: http://localhost:6006")

# Launch TensorBoard
subprocess.run([sys.executable, '-m', 'tensorboard.main', f'--logdir={logs_dir}'])
