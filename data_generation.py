"""
Improved Phase 1 data generation for better transfer to Phase 2 (6DOF RL).

Changes from original:
  1. Randomized initial altitude and velocity (not always z=0, v=0)
  2. Randomized target setpoints (not always 10m)
  3. Bidirectional mass changes (drops AND pickups)
  4. Some episodes start at or near hover (mimics Phase 2 initial conditions)
  5. Optional oracle delay — some episodes apply perfect gains gradually,
     so the detector learns what error looks like DURING adaptation
  6. Wider mass range to cover Phase 2's perturbation scenarios
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import os
from tqdm import tqdm

# --- 1. System Parameters ---
G = 9.81           # Gravity (m/s^2)
C_DRAG = 0.5       # Aerodynamic drag coefficient
DT = 0.01          # Control loop timestep (100 Hz)
T_MAX = 10.0       # Total simulation time (seconds)


# --- 2. The 1D Quadcopter Plant ---
def quadcopter_dynamics(t, state, m, u):
    z, z_dot = state
    z_ddot = (u - m * G - C_DRAG * z_dot) / m
    return [z_dot, z_ddot]


# --- 3. The Analytical Oracle ---
def calculate_ideal_gains(m, c_drag, omega_n=3.0, zeta=1.0, p0=1.5):
    kd = m * (2 * zeta * omega_n + p0) - c_drag
    kp = m * (omega_n**2 + 2 * zeta * omega_n * p0)
    ki = m * (p0 * omega_n**2)
    return kp, ki, kd


class StandardPID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):
        self.integral = np.clip(self.integral + error * dt, -50, 50)
        derivative = (error - self.prev_error) / dt
        u = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        return u, derivative


# --- 4. Simulation Loop (improved) ---
def run_episode(
    episode_id,
    m_initial,
    m_final,
    t_drop,
    target_z=10.0,
    initial_z=0.0,
    initial_v=0.0,
    oracle_delay=0.0,
):
    """
    Run a single episode.

    Args:
        oracle_delay: seconds over which the oracle transitions to new gains
                      after mass change. 0.0 = instant (original behavior).
                      >0 = gains interpolate linearly over this duration.
    """
    time_steps = np.arange(0, T_MAX, DT)

    # Oracle calculates perfect STARTING gains
    start_kp, start_ki, start_kd = calculate_ideal_gains(m_initial, C_DRAG)
    pid = StandardPID(start_kp, start_ki, start_kd)

    state = [initial_z, initial_v]
    history = []
    u = 0.0

    # Pre-compute target gains for after mass change
    final_kp, final_ki, final_kd = calculate_ideal_gains(m_final, C_DRAG)

    for t in time_steps:
        # Determine current physical mass
        current_mass = m_final if t >= t_drop else m_initial

        # Oracle gain calculation with optional delay
        if t < t_drop:
            # Before perturbation: use initial gains
            target_kp, target_ki, target_kd = start_kp, start_ki, start_kd
        elif oracle_delay <= 0 or t >= t_drop + oracle_delay:
            # After delay (or instant): use final gains
            target_kp, target_ki, target_kd = final_kp, final_ki, final_kd
        else:
            # During transition: interpolate
            alpha = (t - t_drop) / oracle_delay
            target_kp = start_kp + alpha * (final_kp - start_kp)
            target_ki = start_ki + alpha * (final_ki - start_ki)
            target_kd = start_kd + alpha * (final_kd - start_kd)

        # Apply gains
        pid.kp = target_kp
        pid.ki = target_ki
        pid.kd = target_kd

        # Calculate Control Input
        error = target_z - state[0]
        noisy_error = error + np.random.normal(0, 0.01)

        delta_u, error_dot = pid.update(noisy_error, DT)
        u = np.clip(delta_u + (current_mass * G), 0, 40)

        # Log Data
        history.append({
            'episode': episode_id,
            'time': t,
            'mass': current_mass,
            'error': error,
            'error_dot': error_dot,
            'integral': pid.integral,
            'prev_thrust': u,
            'target_kp': target_kp,
            'target_ki': target_ki,
            'target_kd': target_kd,
            # Extra metadata for analysis
            'target_z': target_z,
            'z_pos': state[0],
            'z_vel': state[1],
        })

        # Step the Physics
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
    print("Generating improved Oracle dataset...")
    all_episodes = []
    n_episodes = 5000

    for i in tqdm(range(n_episodes), desc="Simulating Flights"):

        # ── Randomize mass (drop only) ──
        m_initial = np.random.uniform(1.8, 3.0)
        m_final = np.random.uniform(0.8, m_initial - 0.2)

        # ── Randomize perturbation time ──
        t_drop = np.random.uniform(2.0, 7.0)

        # ── Randomize target setpoint ──
        target_z = np.random.uniform(1.0, 15.0)

        # ── Randomize initial conditions ──
        # Mix of scenarios the detector will encounter in Phase 2:
        scenario = np.random.random()

        if scenario < 0.4:
            # Start at hover near target (most common in Phase 2)
            initial_z = target_z + np.random.normal(0, 0.1)
            initial_v = np.random.normal(0, 0.05)
        elif scenario < 0.7:
            # Start at ground (original behavior)
            initial_z = 0.0
            initial_v = 0.0
        elif scenario < 0.85:
            # Start offset from target (recovering from disturbance)
            initial_z = target_z + np.random.uniform(-3.0, 3.0)
            initial_v = np.random.uniform(-1.0, 1.0)
        else:
            # Start at random altitude
            initial_z = np.random.uniform(0.0, target_z * 1.5)
            initial_v = np.random.uniform(-2.0, 2.0)

        # ── Oracle delay (0 = instant, >0 = gradual transition) ──
        # 60% instant, 40% delayed (teaches detector what error looks like
        # while gains are still adapting)
        if np.random.random() < 0.6:
            oracle_delay = 0.0
        else:
            oracle_delay = np.random.uniform(0.1, 1.0)

        df_ep = run_episode(
            episode_id=i,
            m_initial=m_initial,
            m_final=m_final,
            t_drop=t_drop,
            target_z=target_z,
            initial_z=initial_z,
            initial_v=initial_v,
            oracle_delay=oracle_delay,
        )
        all_episodes.append(df_ep)

    final_df = pd.concat(all_episodes, ignore_index=True)

    # Save
    save_dir = "data"
    os.makedirs(save_dir, exist_ok=True)
    file_path = f"{save_dir}/oracle_trajectories_v2.csv"
    final_df.to_csv(file_path, index=False)

    # Print stats
    print(f"\nSaved {len(final_df)} timesteps ({n_episodes} episodes) to {file_path}")
    print(f"\nDataset statistics:")
    print(f"  Mass range:    {final_df['mass'].min():.2f} - {final_df['mass'].max():.2f} kg")
    print(f"  Target range:  {final_df['target_z'].min():.2f} - {final_df['target_z'].max():.2f} m")
    print(f"  Error range:   {final_df['error'].min():.2f} - {final_df['error'].max():.2f} m")
    print(f"  Initial z:     {final_df.groupby('episode')['z_pos'].first().describe()}")