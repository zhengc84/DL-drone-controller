import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Load the dataset
file_path = "data/oracle_trajectories.csv"
if not os.path.exists(file_path):
    print(f"Error: Could not find {file_path}. Did you run the generation script?")
    exit()

df = pd.read_csv(file_path)

# 2. Extract a single episode (e.g., Episode 0)
episode_id = 0
ep_data = df[df['episode'] == episode_id]

# 3. Create the plots
fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig.suptitle(f"Oracle Trajectory Visualization (Episode {episode_id})", fontsize=16)

# Plot 1: Mass Drop
axs[0].plot(ep_data['time'], ep_data['mass'], color='red', linewidth=2)
axs[0].set_ylabel("Mass (kg)")
axs[0].grid(True, linestyle='--', alpha=0.7)
axs[0].set_title("System Mass (The Drop)")

# Plot 2: Tracking Error
axs[1].plot(ep_data['time'], ep_data['error'], color='blue', label='Error (z_target - z)')
axs[1].set_ylabel("Error (m)")
axs[1].grid(True, linestyle='--', alpha=0.7)
axs[1].legend()
axs[1].set_title("Altitude Tracking Error")

# Plot 3: Control Input (Thrust)
axs[2].plot(ep_data['time'], ep_data['prev_thrust'], color='green')
axs[2].set_ylabel("Thrust Input (u)")
axs[2].grid(True, linestyle='--', alpha=0.7)
axs[2].set_title("Total Thrust Applied")

# Plot 4: Oracle PID Gains (The Labels for Mamba/LSTM)
axs[3].plot(ep_data['time'], ep_data['target_kp'], label='Kp', color='purple')
axs[3].plot(ep_data['time'], ep_data['target_ki'], label='Ki', color='orange')
axs[3].plot(ep_data['time'], ep_data['target_kd'], label='Kd', color='brown')
axs[3].set_ylabel("Gain Values")
axs[3].set_xlabel("Time (seconds)")
axs[3].grid(True, linestyle='--', alpha=0.7)
axs[3].legend()
axs[3].set_title("Oracle Target Labels (Gain Scheduling)")

plt.tight_layout()
plt.subplots_adjust(top=0.92) # Adjust for the main title
plt.show()