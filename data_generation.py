import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import os
from tqdm import tqdm

# --- 1. System Parameters ---
G = 9.81           # Gravity (m/s^2)
C_DRAG = 0.1       # Aerodynamic drag coefficient
DT = 0.01          # Control loop timestep (100 Hz)
T_MAX = 10.0       # Total simulation time (seconds)

# --- 2. The 1D Quadcopter Plant ---
def quadcopter_dynamics(t, state, m, u):
    """
    Computes the derivatives of the state [z, z_dot].
    Equation: m * z_ddot = u - m*g - c*z_dot
    """
    z, z_dot = state
    z_ddot = (u - m * G - C_DRAG * z_dot) / m
    return [z_dot, z_ddot]

# --- 3. The Oracle PID Controller ---
class OraclePID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        
    def update(self, error, dt):
        self.integral += error * dt
        
        # Anti-windup (optional but recommended for realistic data)
        self.integral = np.clip(self.integral, -50, 50)
        
        derivative = (error - self.prev_error) / dt
        u = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        self.prev_error = error
        return u, derivative

# --- 4. Simulation Loop ---
def run_episode(episode_id, m_initial, m_final, t_drop, target_z=10.0):
    time_steps = np.arange(0, T_MAX, DT)
    
    # 1. Define the mathematical baseline (The "Reference")
    REFERENCE_MASS = 2.0 
    REF_KP, REF_KI, REF_KD = 15.0, 5.0, 10.0 
    
    # 2. Oracle calculates the perfect starting gains for this specific flight
    start_ratio = m_initial / REFERENCE_MASS
    start_kp = REF_KP * start_ratio
    start_ki = REF_KI * start_ratio
    start_kd = REF_KD * start_ratio
    
    pid = OraclePID(start_kp, start_ki, start_kd)
    state = [0.0, 0.0]
    history = []
    u = 0.0 
    
    for t in time_steps:
        current_mass = m_final if t >= t_drop else m_initial
        
        # 3. Oracle continuously scales gains based on the REFERENCE, not the initial
        mass_ratio = current_mass / REFERENCE_MASS
        pid.kp = REF_KP * mass_ratio
        pid.ki = REF_KI * mass_ratio
        pid.kd = REF_KD * mass_ratio
        
        # 2. Calculate Control Input (PID)
        error = target_z - state[0]
        # Add a tiny bit of sensor noise for robust training
        noisy_error = error + np.random.normal(0, 0.01) 
        
        delta_u, error_dot = pid.update(noisy_error, DT)
        
        # Feedforward term to counteract gravity smoothly
        u = delta_u + (current_mass * G) 
        u = np.clip(u, 0, 40) # Simulate motor thrust limits
        
        # 3. Log Data (Features -> Labels)
        history.append({
            'episode': episode_id,
            'time': t,
            'mass': current_mass,
            'error': error,
            'error_dot': error_dot,
            'integral': pid.integral,
            'prev_thrust': u,
            'target_kp': pid.kp,  # Label Y1
            'target_ki': pid.ki,  # Label Y2
            'target_kd': pid.kd   # Label Y3
        })
        
        # 4. Step the Physical Plant Forward
        sol = solve_ivp(
            quadcopter_dynamics, 
            [t, t + DT], 
            state, 
            args=(current_mass, u),
            method='RK45'
        )
        state = [sol.y[0][-1], sol.y[1][-1]]
        
    return pd.DataFrame(history)

# --- 5. Generate Dataset ---
if __name__ == "__main__":
    print("Generating Oracle dataset...")
    all_episodes = []
    
    # Generate 100 randomized episodes for a quick test
    for i in tqdm(range(5000), desc="Simulating Flights"):
        # Randomize takeoff weight between 1.8kg and 3.0kg
        m_initial_random = np.random.uniform(1.8, 3.0)
        
        # Randomize how much is dropped (leaving between 0.8kg and 1.5kg)
        # Ensure m_final is always strictly less than m_initial!
        m_final_random = np.random.uniform(0.8, m_initial_random - 0.2) 
        
        t_drop_random = np.random.uniform(3.0, 7.0)
        
        df_ep = run_episode(i, m_initial_random, m_final_random, t_drop_random)
        all_episodes.append(df_ep)
        
    final_df = pd.concat(all_episodes, ignore_index=True)
    
    # Save the dataset into your project structure
    save_dir = "data"
    os.makedirs(save_dir, exist_ok=True)
    file_path = f"{save_dir}/oracle_trajectories.csv"
    
    final_df.to_csv(file_path, index=False)
    print(f"Saved {len(final_df)} timesteps to {file_path}.")