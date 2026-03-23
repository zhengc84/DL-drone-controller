import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import os

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
def run_episode(episode_id, m_initial=2.0, m_final=1.2, t_drop=5.0, target_z=10.0):
    time_steps = np.arange(0, T_MAX, DT)
    
    # Base gains tuned for the initial mass
    base_kp, base_ki, base_kd = 15.0, 5.0, 10.0 
    pid = OraclePID(base_kp, base_ki, base_kd)
    
    # State tracking
    state = [0.0, 0.0]  # [z, z_dot] starting on the ground
    history = []
    
    u = 0.0 # Initial thrust
    
    for t in time_steps:
        # 1. Oracle checks mass and adjusts gains perfectly
        current_mass = m_final if t >= t_drop else m_initial
        
        # Oracle scales gains proportional to mass to maintain identical dynamics
        mass_ratio = current_mass / m_initial
        pid.kp = base_kp * mass_ratio
        pid.ki = base_ki * mass_ratio
        pid.kd = base_kd * mass_ratio
        
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
    for i in range(100):
        t_drop_random = np.random.uniform(3.0, 7.0)
        m_final_random = np.random.uniform(0.8, 1.5)
        
        df_ep = run_episode(i, t_drop=t_drop_random, m_final=m_final_random)
        all_episodes.append(df_ep)
        
    final_df = pd.concat(all_episodes, ignore_index=True)
    
    # Save the dataset into your project structure
    save_dir = "data"
    os.makedirs(save_dir, exist_ok=True)
    file_path = f"{save_dir}/oracle_trajectories.csv"
    
    final_df.to_csv(file_path, index=False)
    print(f"Saved {len(final_df)} timesteps to {file_path}.")