"""
Run a single episode for each trained model and plot:
  - Drone z-position vs target
  - Tracking error
  - Z-axis PID gains over time

Usage:
    python demo_flight.py

Edit the MODELS dict below to point to your checkpoint paths.
"""

import torch
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from config import Config
from envs import AdaptivePIDEnv
from lstm_policy import LSTMPolicy
from mamba_policy import MambaPolicy
from pretrained_policy import PretrainedRLPolicy


def load_model(model_type, checkpoint_path, obs_dim, cfg, detector_info=None):
    """Load any of the four model types."""
    mc = cfg.model

    if model_type == "lstm":
        policy = LSTMPolicy(
            obs_dim=obs_dim,
            action_dim=mc.action_dim,
            hidden_size=mc.hidden_size,
            num_layers=mc.num_layers,
        )
    elif model_type == "mamba":
        policy = MambaPolicy(
            obs_dim=obs_dim,
            action_dim=mc.action_dim,
            hidden_size=mc.hidden_size,
            num_layers=mc.num_layers,
            d_state=mc.mamba_d_state,
            d_conv=mc.mamba_d_conv,
            expand=mc.mamba_expand,
        )
    elif model_type.startswith("pretrained"):
        det_type = detector_info["detector_type"]
        det_weights = detector_info["detector_weights"]
        policy = PretrainedRLPolicy(
            obs_dim=obs_dim,
            action_dim=mc.action_dim,
            detector_type=det_type,
            detector_weights=det_weights,
            hidden_size=mc.hidden_size,
            error_history_len=cfg.env.error_history_len,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()
    return policy


def run_episode(env, policy, device="cpu", seed=42):
    """Run one deterministic episode, collecting data for plotting."""
    np.random.seed(seed)
    obs, info = env.reset()
    hidden = policy.get_initial_hidden(1, device)
    done = False

    data = {
        "time": [],
        "z_pos": [],
        "z_target": [],
        "error_norm": [],
        "kp_z": [],
        "ki_z": [],
        "kd_z": [],
        "mass": [],
        "perturb_time": None,
    }

    ctrl_dt = 1.0 / env.cfg.ctrl_freq
    gain_dt = ctrl_dt * env.cfg.gain_update_interval

    while not done:
        t = env.step_count * ctrl_dt

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _, _, hidden = policy(obs_t, hidden)
        action_np = action.squeeze(0).cpu().numpy()
        if action_np.ndim > 1:
            action_np = action_np.squeeze(0)

        # Use mean action (no exploration noise)
        obs, reward, terminated, truncated, step_info = env.step(action_np)
        done = terminated or truncated

        data["time"].append(t + gain_dt)  # time after this decision
        data["z_pos"].append(step_info["position"][2])
        data["z_target"].append(step_info["target"][2])
        data["error_norm"].append(step_info["error_norm"])
        data["kp_z"].append(step_info["pid_gains"][2])
        data["ki_z"].append(step_info["pid_gains"][5])
        data["kd_z"].append(step_info["pid_gains"][8])
        data["mass"].append(step_info["mass"])

        if step_info["mass_perturbed"] and data["perturb_time"] is None:
            data["perturb_time"] = t

    return data


def run_baseline(env, seed=42):
    """Run one episode with default PID (no agent)."""
    np.random.seed(seed)
    obs, info = env.reset()
    done = False

    data = {
        "time": [],
        "z_pos": [],
        "z_target": [],
        "error_norm": [],
        "kp_z": [],
        "ki_z": [],
        "kd_z": [],
        "mass": [],
        "perturb_time": None,
    }

    ctrl_dt = 1.0 / env.cfg.ctrl_freq
    gain_dt = ctrl_dt * env.cfg.gain_update_interval

    while not done:
        t = env.step_count * ctrl_dt
        action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated

        data["time"].append(t + gain_dt)
        data["z_pos"].append(step_info["position"][2])
        data["z_target"].append(step_info["target"][2])
        data["error_norm"].append(step_info["error_norm"])
        data["kp_z"].append(step_info["pid_gains"][2])
        data["ki_z"].append(step_info["pid_gains"][5])
        data["kd_z"].append(step_info["pid_gains"][8])
        data["mass"].append(step_info["mass"])

        if step_info["mass_perturbed"] and data["perturb_time"] is None:
            data["perturb_time"] = t

    return data


def plot_comparison(all_data, save_path="demo_flight.png"):
    """Plot all models side by side."""
    colors = {
        "Baseline": "#888888",
        "LSTM": "#2196F3",
        "MAMBA": "#FF5722",
        "Pretrained LSTM": "#4CAF50",
        "Pretrained MAMBA": "#9C27B0",
    }

    fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=True)

    # Find a common perturb time for the vertical line
    perturb_time = None
    for name, data in all_data.items():
        if data["perturb_time"] is not None:
            perturb_time = data["perturb_time"]
            break

    for name, data in all_data.items():
        t = data["time"]
        c = colors.get(name, "#333333")

        # Row 1: Z position
        axes[0].plot(t, data["z_pos"], label=name, color=c, linewidth=1.5)

        # Row 2: Tracking error
        axes[1].plot(t, data["error_norm"], label=name, color=c, linewidth=1.5)

        # Row 3: Kp_z
        axes[2].plot(t, data["kp_z"], label=f"{name} Kp_z", color=c,
                    linewidth=1.5, linestyle="-")

        # Row 4: Ki_z
        axes[3].plot(t, data["ki_z"], label=f"{name} Ki_z", color=c,
                    linewidth=1.5, linestyle="-")

        # Row 5: Kd_z
        axes[4].plot(t, data["kd_z"], label=f"{name} Kd_z", color=c,
                    linewidth=1.5, linestyle="-")

    # Add target line and perturbation marker
    if all_data:
        first = next(iter(all_data.values()))
        axes[0].plot(first["time"], first["z_target"], "r--", alpha=0.5,
                    label="Target", linewidth=1)

    for ax in axes:
        if perturb_time is not None:
            ax.axvline(perturb_time, color="red", linestyle=":", linewidth=2,
                      alpha=0.7, label="Mass drop")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[0].set_ylabel("Z Position (m)")
    axes[0].set_title("Drone Flight: Model Comparison", fontsize=14, fontweight="bold")

    axes[1].set_ylabel("Error (m)")
    axes[1].axhline(0.05, color="gray", linestyle="--", alpha=0.5, label="5cm tolerance")

    axes[2].set_ylabel("Kp_z")

    axes[3].set_ylabel("Ki_z")

    axes[4].set_ylabel("Kd_z")
    axes[4].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {save_path}")


def main():
    import glob

    cfg = Config()
    cfg.env.hover_height = 5.0  # start at 2m instead of 1m
    cfg.env.mass_delta_range = (0.012, 0.016)  # 44-59% drop
    cfg.env.episode_len_sec = 15.0             # longer episode to see full recovery
    cfg.env.perturb_time_range = (3.0, 4.0)    # consistent drop time
    env = AdaptivePIDEnv(config=cfg.env)
    obs_dim = env.observation_space.shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Find model checkpoints ──
    # Auto-detect from logs directory
    all_data = {}

    # 1. Baseline (no agent)
    print("Running baseline...")
    all_data["Baseline"] = run_baseline(env, seed=99)

    # 2. Find all runs and load best model from each
    run_dirs = sorted(glob.glob("logs/*"))
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        config_path = run_dir / "config.json"
        best_model_path = run_dir / "best_model.pt"

        if not config_path.exists() or not best_model_path.exists():
            continue

        with open(config_path) as f:
            run_config = json.load(f)

        model_type = run_config.get("model_type", "unknown")

        # Build display name
        if model_type == "pretrained":
            det_type = run_config.get("detector_type", "unknown")
            display_name = f"Pretrained {det_type.upper()}"
            detector_info = {
                "detector_type": det_type,
                "detector_weights": run_config.get("detector_weights"),
            }
        else:
            display_name = model_type.upper()
            detector_info = None

        if display_name in all_data:
            # Skip duplicate model types (pick first run found)
            continue

        print(f"Loading {display_name} from {run_dir}...")
        try:
            policy = load_model(
                model_type, str(best_model_path), obs_dim, cfg,
                detector_info=detector_info,
            ).to(device)
            print(f"Running {display_name}...")
            all_data[display_name] = run_episode(env, policy, device, seed=99)
        except Exception as e:
            print(f"  Failed: {e}")

    env.close()

    if len(all_data) < 2:
        print("Need at least baseline + one model. Check logs/ directory.")
        return

    plot_comparison(all_data, save_path="demo_flight.png")
    print(f"\nPlotted {len(all_data)} models.")


if __name__ == "__main__":
    main()