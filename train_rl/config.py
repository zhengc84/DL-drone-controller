"""Centralized configuration for the adaptive PID RL project."""

from dataclasses import dataclass, field
from typing import List, Tuple
import torch


@dataclass
class EnvConfig:
    """Environment configuration."""
    # Drone physics
    drone_model: str = "cf2x"  # Crazyflie 2.x
    freq: int = 240  # Simulation frequency (Hz)
    ctrl_freq: int = 48  # Control frequency (Hz)
    episode_len_sec: float = 8.0  # Episode duration

    # Mass perturbation
    nominal_mass_kg: float = 0.027  # CF2x nominal mass
    mass_delta_range: Tuple[float, float] = (0.012, 0.022)  # kg added/removed
    perturb_time_range: Tuple[float, float] = (2.0, 3.0)  # when perturbation occurs (sec)
    drop_probability: float = 1.0  # 1.0 = always drop, 0.0 = always pickup

    # Target trajectory
    hover_height: float = 1.0  # meters
    target_type: str = "hover"  # "hover", "step", "sinusoid"

    # PID gain bounds (for action clipping)
    kp_range: Tuple[float, float] = (0.1, 20.0)
    ki_range: Tuple[float, float] = (0.0, 10.0)
    kd_range: Tuple[float, float] = (0.0, 10.0)

    gain_update_interval: int = 24  # timesteps between PID gain updates (matches ctrl_freq)

    # Observation
    error_history_len: int = 50  # timesteps of error history to keep
    include_pid_gains_in_obs: bool = True

    # Reward shaping (needed by env)
    reward_type: str = "iae"  # "ise", "itae", "iae"
    settling_bonus: float = 5.0
    settling_tolerance: float = 0.05  # meters
    overshoot_penalty: float = 2.0
    control_effort_penalty: float = 0.01


@dataclass
class ModelConfig:
    """Sequence model configuration."""
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.0

    # Mamba-specific
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2

    # Action head
    action_dim: int = 3  # ΔKp, ΔKi, ΔKd for x, y, z axes
    log_std_init: float = -1.0


@dataclass
class TrainConfig:
    """PPO training configuration."""
    # PPO hyperparams
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 0.25
    max_grad_norm: float = 0.5

    # Normalization
    normalize_rewards: bool = True
    reward_clip: float = 10.0

    # Training schedule
    total_episodes: int = 5000
    steps_per_update: int = 2048
    num_epochs: int = 10
    minibatch_size: int = 64
    num_envs: int = 4  # parallel envs

    # Sequence training
    seq_len: int = 50  # BPTT sequence length (matches error_history_len)

    # Logging
    log_interval: int = 10
    save_interval: int = 100
    eval_episodes: int = 100
    log_dir: str = "logs"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
