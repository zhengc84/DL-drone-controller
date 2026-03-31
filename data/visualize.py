import pandas as pd
import matplotlib.pyplot as plt

def plot_pristine_episode(csv_path="data/oracle_trajectories.csv", episode_id=1):
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # --- THE CLEANUP ---
    # This deletes the very first row of every episode where the 
    # math simulation boots up and the velocity spikes to 1000.
    df = df[df.groupby('episode').cumcount() > 0].reset_index(drop=True)
    # -------------------
    
    # Grab all the data for the requested episode
    flight_data = df[df['episode'] == episode_id].reset_index(drop=True)
    
    if flight_data.empty:
        print(f"Episode {episode_id} not found!")
        return
        
    # Create the time axis (assuming DT = 0.01 seconds)
    time_axis = flight_data.index * 0.01
    
    # Set up a beautiful dark-mode plot
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # --- Top Graph: The Telemetry (Inputs) ---
    ax1.plot(time_axis, flight_data['error'], label='Error (m)', color='#00ff00', linewidth=2)
    ax1.plot(time_axis, flight_data['error_dot'], label='Error_Dot (Velocity)', color='#ff00ff', alpha=0.7)
    ax1.plot(time_axis, flight_data['integral'], label='Integral', color='#00ffff', alpha=0.5)
    
    ax1.set_title(f"Episode {episode_id}: Pristine Drone Telemetry", fontsize=14)
    ax1.set_ylabel("Sensor Values")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.2)
    
    # --- Bottom Graph: The Oracle Targets (Outputs) ---
    ax2.plot(time_axis, flight_data['target_kp'], label='Target Kp', color='#ff7f0e', linewidth=2)
    ax2.plot(time_axis, flight_data['target_ki'], label='Target Ki', color='#1f77b4', linewidth=2)
    ax2.plot(time_axis, flight_data['target_kd'], label='Target Kd', color='#d62728', linewidth=2)
    
    ax2.set_title("Oracle PID Gains", fontsize=14)
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Gain Values")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(f"episode_{episode_id}_pristine.png", dpi=300)
    print(f"Saved pristine visualization to episode_{episode_id}_pristine.png")

if __name__ == "__main__":
    # Feel free to change this number to look at different flights!
    plot_pristine_episode(episode_id=1)