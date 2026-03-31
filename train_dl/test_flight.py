import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp

# Import the architectures we defined in your training script
from train import BaselineLSTM, MambaPIDTuner
from dataset import DroneSequenceDataset

# --- 1. Physics Constants ---
G = 9.81
C_DRAG = 0.5  # The realistic aerodynamic drag we discussed
DT = 0.01     # 100Hz control loop
T_MAX = 10.0  # 10 second flight
DROP_TIME = 5.0

# --- 2. Physics Engine ---
def quadcopter_dynamics(t, state, m, u):
    z, z_dot = state
    z_ddot = (u - m * G - C_DRAG * z_dot) / m
    return [z_dot, z_ddot]

# --- 3. The Flight Simulator ---
def run_flight(model, model_name, data_mean, data_std, device='cuda'):
    print(f"\nBooting physical simulation for: {model_name}...")
    model.eval()
    
    # Starting conditions
    m_initial = 3.0
    m_final = 1.5
    target_z = 10.0
    
    state = [10.0, 0.0]  # [z, z_dot]
    integral = 0.0
    prev_error = target_z
    prev_thrust = m_initial * G
    
    # We need a 50-step rolling window to feed the models
    # Initialize it with zeros to represent the drone sitting on the ground
    perfect_hover_state = np.array([0.0, 0.0, 0.0, m_initial * G])
    
    # Copy that exact state 50 times to fill the history buffer
    history_buffer = np.tile(perfect_hover_state, (50, 1))
    
    time_steps = np.arange(0, T_MAX, DT)
    altitude_log = []
    kp_log, ki_log, kd_log = [], [], []
    inference_times = []
    
    for t in time_steps:
        # 1. Trigger the step change in mass
        current_mass = m_final if t >= DROP_TIME else m_initial
        
        # 2. Update physical sensors
        error = target_z - state[0]
        error_dot = (error - prev_error) / DT
        integral = np.clip(integral + error * DT, -50, 50)
        
        # 3. Update rolling window (pop oldest, append newest)
        current_step_data = np.array([error, error_dot, integral, prev_thrust])
        history_buffer = np.roll(history_buffer, -1, axis=0)
        history_buffer[-1] = current_step_data
        
        # 4. Normalize the window using the training statistics
        norm_window = (history_buffer - data_mean) / data_std
        
        # 5. Neural Network Inference (Timing it for the KPI)
        start_time = time.perf_counter()

        if t < 0.5: # For the first 50 steps...
            # Use safe, hardcoded Oracle gains for a 3.0kg drone to generate real physics noise
            predicted_gains = np.array([40.0, 30.0, 18.0]) 
            inference_times.append(0) # No inference time
        else:
            # At 0.51s, the buffer is full of real physics. Let the Neural Net take the wheel!
            with torch.no_grad():
                x_tensor = torch.tensor(norm_window, dtype=torch.float32).unsqueeze(0).to(device)
                predicted_gains = model(x_tensor).cpu().numpy()[0]
            inference_times.append((time.perf_counter() - start_time) * 1000)
        
        kp, ki, kd = predicted_gains
        
        # 6. Apply predicted gains to the PID controller
        delta_u = (kp * error) + (ki * integral) + (kd * error_dot)
        thrust = np.clip(delta_u + (m_initial * G), 0, 40)
        
        # 7. Step the physics forward
        sol = solve_ivp(quadcopter_dynamics, [t, t + DT], state, args=(current_mass, thrust), method='RK45')
        state = [sol.y[0][-1], sol.y[1][-1]]
        
        # Update trackers
        # Update trackers
        prev_error = error
        prev_thrust = thrust
        altitude_log.append(state[0])
        kp_log.append(kp)
        ki_log.append(ki) # --- ADD THIS ---
        kd_log.append(kd) # --- ADD THIS ---
        
    avg_latency = np.mean(inference_times)
    print(f"[{model_name}] Flight Complete. Avg Latency: {avg_latency:.2f} ms")
    
    # --- UPDATE THIS RETURN STATEMENT ---
    return time_steps, altitude_log, kp_log, ki_log, kd_log, avg_latency

# --- 4. Execution & Plotting ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Grab the exact normalization stats used during training
    dataset = DroneSequenceDataset("data/oracle_trajectories_v2.csv", sequence_length=50)
    data_mean = dataset.mean
    data_std = dataset.std
    
    # 2. Load the trained models
    lstm = BaselineLSTM().to(device)
    lstm.load_state_dict(torch.load("LSTM_Tuner_weights.pth", map_location=device))
    
    mamba = MambaPIDTuner().to(device)
    mamba.load_state_dict(torch.load("Mamba_Tuner_weights.pth", map_location=device))
    
    # 3. Run the flight tests (Unpacking all the new gain logs!)
    t, alt_lstm, kp_lstm, ki_lstm, kd_lstm, lat_lstm = run_flight(lstm, "LSTM Controller", data_mean, data_std, device)
    _, alt_mamba, kp_mamba, ki_mamba, kd_mamba, lat_mamba = run_flight(mamba, "Mamba Controller", data_mean, data_std, device)
    
    # 4. Plot the Altitude AND the PID Gains
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- TOP GRAPH: Altitude ---
    ax1.plot(t, [10.0]*len(t), 'w--', alpha=0.5, label="Target (10.0m)")
    ax1.axvline(x=DROP_TIME, color='r', linestyle=':', label="Payload Drop")
    
    ax1.plot(t, alt_lstm, label=f"LSTM Altitude", color='#1f77b4', linewidth=2)
    ax1.plot(t, alt_mamba, label=f"Mamba Altitude", color='#ff7f0e', linewidth=2)
    
    ax1.set_title("Flight Performance & Neural Network PID Predictions", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Altitude (meters)", fontsize=12)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.2)
    
    # 4. Plot the Altitude AND ALL PID Gains
    plt.style.use('dark_background')
    # Upgraded to a 4-row subplot to fit everything cleanly
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # --- ROW 1: Altitude ---
    ax1.plot(t, [10.0]*len(t), 'w--', alpha=0.5, label="Target (10.0m)")
    ax1.axvline(x=DROP_TIME, color='r', linestyle=':', label="Payload Drop")
    ax1.plot(t, alt_lstm, label=f"LSTM Altitude", color='#1f77b4', linewidth=2)
    ax1.plot(t, alt_mamba, label=f"Mamba Altitude", color='#ff7f0e', linewidth=2)
    
    ax1.set_title("Flight Performance & Full PID Predictions", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Altitude (m)", fontsize=12)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.2)
    
    # --- ROW 2: Kp Gain ---
    ax2.axvline(x=DROP_TIME, color='r', linestyle=':')
    ax2.plot(t, kp_lstm, label="LSTM Kp", color='#1f77b4', linestyle='--')
    ax2.plot(t, kp_mamba, label="Mamba Kp", color='#ff7f0e', linestyle='--')
    ax2.set_ylabel("Kp (Proportional)", fontsize=12)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.2)
    
    # --- ROW 3: Ki Gain ---
    ax3.axvline(x=DROP_TIME, color='r', linestyle=':')
    ax3.plot(t, ki_lstm, label="LSTM Ki", color='#1f77b4', linestyle='-.')
    ax3.plot(t, ki_mamba, label="Mamba Ki", color='#ff7f0e', linestyle='-.')
    ax3.set_ylabel("Ki (Integral)", fontsize=12)
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.2)
    
    # --- ROW 4: Kd Gain ---
    ax4.axvline(x=DROP_TIME, color='r', linestyle=':')
    ax4.plot(t, kd_lstm, label="LSTM Kd", color='#1f77b4', linestyle=':')
    ax4.plot(t, kd_mamba, label="Mamba Kd", color='#ff7f0e', linestyle=':')
    ax4.set_xlabel("Time (seconds)", fontsize=12)
    ax4.set_ylabel("Kd (Derivative)", fontsize=12)
    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig("full_flight_dashboard.png", dpi=300)
    print("\nGraph saved as 'full_flight_dashboard.png'.")