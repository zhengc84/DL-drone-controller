"""
Adaptive PID Environment

Wraps gym-pybullet-drones' CtrlAviary with:
  - Mid-episode mass perturbation (weight drop/pickup)
  - Action space = PID gain adjustments (ΔKp, ΔKi, ΔKd per axis)
  - Observation = error history + drone state + current PID gains
  - Reward based on tracking performance (ISE/ITAE/IAE)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
from typing import Optional, Tuple, Dict, Any
from collections import deque

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics

import sys
sys.path.append("..")
from config import EnvConfig


class AdaptivePIDEnv(gym.Env):
    """
    RL environment for adaptive PID tuning under mass perturbation.

    Observation space:
        - Position error history (error_history_len × 3)
        - Current drone state: pos(3), vel(3), rpy(3), angular_vel(3)
        - Current PID gains: Kp(3), Ki(3), Kd(3)
        - Mass perturbation flag (1) — 0 before, 1 after perturbation
        - Time in episode (1)

    Action space:
        - ΔKp(3), ΔKi(3), ΔKd(3) — continuous adjustments to PID gains

    Reward:
        - Negative tracking error (ISE/ITAE/IAE) + settling bonus - control effort penalty
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode

        # Timing
        self.ctrl_dt = 1.0 / self.cfg.ctrl_freq
        self.sim_steps_per_ctrl = self.cfg.freq // self.cfg.ctrl_freq
        self.max_steps = int(self.cfg.episode_len_sec * self.cfg.ctrl_freq)

        # --- Action space: z-axis PID gains only (Kp_z, Ki_z, Kd_z) ---
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(3,), dtype=np.float32,
        )

        # Default CF2x gains from DSLPIDControl
        self.default_gains = np.array([
            0.4, 0.4, 1.25,    # Kp (x, y, z)
            0.05, 0.05, 0.05,  # Ki
            0.2, 0.2, 0.5,     # Kd
        ], dtype=np.float32)

        # Z-axis gain mapping: action=0 → default, action=±1 → ± range
        self.z_gain_mid = np.array([1.25, 0.05, 0.5], dtype=np.float32)
        self.z_gain_half_range = np.array([1.0, 0.05, 0.45], dtype=np.float32)

        # --- Observation space ---
        error_hist_dim = self.cfg.error_history_len * 3
        state_dim = 12  # pos(3) + vel(3) + rpy(3) + ang_vel(3)
        pid_dim = 3     # Kp(3) + Ki(3) + Kd(3)
        extra_dim = 2   # perturbation flag + normalized time
        obs_dim = error_hist_dim + state_dim + pid_dim + extra_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32,
        )

        # Internal state (initialized in reset)
        self.aviary = None
        self.pid_ctrl = None
        self.error_history = None
        self.current_gains = None
        self.step_count = 0
        self.perturb_step = 0
        self.mass_perturbed = False
        self.target_pos = None
        self.integral_error = np.zeros(3)

    def _get_target(self, t: float) -> np.ndarray:
        """Get target position at time t."""
        if self.cfg.target_type == "hover":
            return np.array([0.0, 0.0, self.cfg.hover_height])
        elif self.cfg.target_type == "step":
            # Step change at t=episode/2
            h = self.cfg.hover_height
            return np.array([0.0, 0.0, h + 0.3 if t > self.cfg.episode_len_sec / 2 else h])
        elif self.cfg.target_type == "sinusoid":
            h = self.cfg.hover_height
            return np.array([
                0.3 * np.sin(0.5 * np.pi * t),
                0.3 * np.cos(0.5 * np.pi * t),
                h + 0.2 * np.sin(0.25 * np.pi * t),
            ])
        return np.array([0.0, 0.0, self.cfg.hover_height])

    def _get_default_pid_gains(self) -> np.ndarray:
        """Return default PID gains from DSLPIDControl as [Kp_xyz, Ki_xyz, Kd_xyz]."""
        # These are the default CF2x gains — you may want to read them
        # directly from self.pid_ctrl if the library exposes them.
        return np.array([
            # Kp for x, y, z
            0.4, 0.4, 1.25,
            # Ki for x, y, z
            0.05, 0.05, 0.05,
            # Kd for x, y, z
            0.2, 0.2, 0.5,
        ], dtype=np.float32)

    def _apply_gains_to_pid(self, gains: np.ndarray):
        """
        Write current gains into the PID controller.

        NOTE: gym-pybullet-drones' DSLPIDControl stores gains as class attributes.
        You'll need to patch these based on the library version you're using.
        Inspect DSLPIDControl.__init__ for the exact attribute names.
        """
        kp = gains[0:3]
        ki = gains[3:6]
        kd = gains[6:9]

        # --- PATCH POINT ---
        # The exact attribute names depend on your gym-pybullet-drones version.
        # Common patterns:
        #   self.pid_ctrl.P_COEFF_FOR = np.diag(kp)  (for position)
        #   self.pid_ctrl.I_COEFF_FOR = np.diag(ki)
        #   self.pid_ctrl.D_COEFF_FOR = np.diag(kd)
        # Inspect your version and update accordingly:
        try:
            self.pid_ctrl.P_COEFF_FOR = np.array([kp[0], kp[1], kp[2]])
            self.pid_ctrl.I_COEFF_FOR = np.array([ki[0], ki[1], ki[2]])
            self.pid_ctrl.D_COEFF_FOR = np.array([kd[0], kd[1], kd[2]])
        except AttributeError:
            # Fallback: store gains and apply manually in compute_control
            self._manual_gains = (kp, ki, kd)

    def _perturb_mass(self):
        """Change drone mass mid-episode via PyBullet."""
        delta = np.random.uniform(*self.cfg.mass_delta_range)
        if np.random.random() < self.cfg.drop_probability:
            delta = -delta  # weight drop

        new_mass = self.cfg.nominal_mass_kg + delta
        new_mass = max(0.01, new_mass)  # floor to prevent negative mass

        # PyBullet: change mass of the drone's base link
        drone_id = self.aviary.DRONE_IDS[0]
        p.changeDynamics(
            drone_id, -1,  # -1 = base link
            mass=new_mass,
            physicsClientId=self.aviary.CLIENT,
        )
        self.mass_perturbed = True
        self._actual_mass = new_mass

    def _compute_reward(self, error: np.ndarray, action: np.ndarray) -> float:
        """Compute reward based on tracking error and control effort."""
        t = self.step_count * self.ctrl_dt
        err_norm = np.linalg.norm(error)

        if self.cfg.reward_type == "itae":
            # Integral of Time-weighted Absolute Error
            tracking_cost = t * err_norm
        elif self.cfg.reward_type == "ise":
            tracking_cost = err_norm ** 2
        else:  # iae
            tracking_cost = err_norm

        reward = -tracking_cost

        # Settling bonus
        if err_norm < self.cfg.settling_tolerance:
            reward += self.cfg.settling_bonus * self.ctrl_dt

        # Control effort penalty (penalize large gain changes)
        reward -= self.cfg.control_effort_penalty * np.sum(action ** 2)

        return float(reward)

    def _build_obs(self, error: np.ndarray) -> np.ndarray:
        """Construct observation vector."""
        # Flatten error history: (history_len, 3) -> (history_len * 3,)
        hist_array = np.array(self.error_history, dtype=np.float32)
        if len(hist_array) < self.cfg.error_history_len:
            pad = np.zeros((self.cfg.error_history_len - len(hist_array), 3), dtype=np.float32)
            hist_array = np.concatenate([pad, hist_array], axis=0)
        hist_flat = hist_array.flatten()

        # Drone state
        state = self.aviary._getDroneStateVector(0)
        pos = state[0:3].astype(np.float32)
        vel = state[10:13].astype(np.float32)
        rpy = state[7:10].astype(np.float32)
        ang_vel = state[13:16].astype(np.float32)
        drone_state = np.concatenate([pos, vel, rpy, ang_vel])

        # Current PID gains (normalized)
        pid_gains = np.array([
            self.current_gains[2],   # Kp_z
            self.current_gains[5],   # Ki_z
            self.current_gains[8],   # Kd_z
        ], dtype=np.float32)

        # Extras
        perturb_flag = np.array([1.0 if self.mass_perturbed else 0.0], dtype=np.float32)
        norm_time = np.array([self.step_count / self.max_steps], dtype=np.float32)

        obs = np.concatenate([hist_flat, drone_state, pid_gains, perturb_flag, norm_time])
        return obs

    def _create_aviary(self):
        """Create the PyBullet simulation (only done once)."""
        gui = self.render_mode == "human"
        self.aviary = CtrlAviary(
            drone_model=DroneModel(self.cfg.drone_model),
            num_drones=1,
            initial_xyzs=np.array([[0.0, 0.0, self.cfg.hover_height]]),
            physics=Physics.PYB,
            pyb_freq=self.cfg.freq,
            ctrl_freq=self.cfg.ctrl_freq,
            gui=gui,
        )
        self.pid_ctrl = DSLPIDControl(drone_model=DroneModel(self.cfg.drone_model))

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Create sim on first call, reuse after that
        if self.aviary is None:
            self._create_aviary()
        else:
            # Reset drone position and velocity via PyBullet directly
            drone_id = self.aviary.DRONE_IDS[0]
            init_pos = [0.0, 0.0, self.cfg.hover_height]
            init_orn = p.getQuaternionFromEuler([0, 0, 0])
            p.resetBasePositionAndOrientation(
                drone_id, init_pos, init_orn,
                physicsClientId=self.aviary.CLIENT,
            )
            p.resetBaseVelocity(
                drone_id, [0, 0, 0], [0, 0, 0],
                physicsClientId=self.aviary.CLIENT,
            )
            # Restore nominal mass
            p.changeDynamics(
                drone_id, -1,
                mass=self.cfg.nominal_mass_kg,
                physicsClientId=self.aviary.CLIENT,
            )

        # Reset PID gains
        self.current_gains = self.default_gains.copy()
        self._apply_gains_to_pid(self.current_gains)

        # Reset tracking
        self.error_history = deque(maxlen=self.cfg.error_history_len)
        self.integral_error = np.zeros(3)
        self.step_count = 0
        self.mass_perturbed = False
        self._actual_mass = self.cfg.nominal_mass_kg

        # Randomize perturbation time
        self.perturb_step = int(
            np.random.uniform(*self.cfg.perturb_time_range) * self.cfg.ctrl_freq
        )

        # Initial observation (zero error)
        init_error = np.zeros(3, dtype=np.float32)
        self.error_history.append(init_error)
        obs = self._build_obs(init_error)

        info = {
            "nominal_mass": self.cfg.nominal_mass_kg,
            "perturb_step": self.perturb_step,
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # --- 1. Update PID gains from action (once per agent decision) ---
        z_gains = self.z_gain_mid + action * self.z_gain_half_range
        self.current_gains = self.default_gains.copy()
        self.current_gains[2] = z_gains[0]   # Kp_z
        self.current_gains[5] = z_gains[1]   # Ki_z
        self.current_gains[8] = z_gains[2]   # Kd_z
        self._apply_gains_to_pid(self.current_gains)

        # --- 2. Run N sim steps with fixed gains ---
        total_reward = 0.0
        terminated = False

        for _ in range(self.cfg.gain_update_interval):
            self.step_count += 1

            # Apply mass perturbation if it's time
            if self.step_count == self.perturb_step and not self.mass_perturbed:
                self._perturb_mass()

            # Compute PID output and step sim
            t = self.step_count * self.ctrl_dt
            target = self._get_target(t)
            state = self.aviary._getDroneStateVector(0)

            rpm, _, _ = self.pid_ctrl.computeControlFromState(
                control_timestep=self.ctrl_dt,
                state=state,
                target_pos=target,
            )

            obs_dict, _, _, _, _ = self.aviary.step(
                np.array([rpm])
            )

            # Compute error
            new_state = self.aviary._getDroneStateVector(0)
            pos = new_state[0:3]
            error = target - pos
            self.error_history.append(error.astype(np.float32))

            # Accumulate reward
            total_reward += self._compute_reward(error, action)

            # Check termination
            if np.linalg.norm(error) > 2.0 or abs(new_state[7]) > np.pi / 2:
                terminated = True
                total_reward -= 50.0
                break

            if self.step_count >= self.max_steps:
                break

        # --- 3. Termination ---
        truncated = self.step_count >= self.max_steps

        # --- 4. Build observation ---
        obs = self._build_obs(error)

        info = {
            "error_norm": float(np.linalg.norm(error)),
            "mass": self._actual_mass,
            "pid_gains": self.current_gains.copy(),
            "mass_perturbed": self.mass_perturbed,
            "position": pos.copy(),
            "target": target.copy(),
        }

        return obs, total_reward, terminated, truncated, info

    def render(self):
        """Rendering handled by PyBullet GUI if render_mode='human'."""
        pass

    def close(self):
        if self.aviary is not None:
            self.aviary.close()
            self.aviary = None


# --- Vectorized env factory ---
def make_adaptive_pid_env(config: EnvConfig, render_mode=None):
    """Factory function for creating the env (compatible with SB3 make_vec_env)."""
    def _init():
        return AdaptivePIDEnv(config=config, render_mode=render_mode)
    return _init