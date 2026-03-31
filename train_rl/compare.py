"""
Compare LSTM vs Mamba training runs.

Generates comparison plots for:
  - Training reward curves
  - Post-perturbation tracking error
  - Settling time after mass change
  - PID gain adaptation trajectories

Usage:
    python compare.py --runs logs/lstm_seed42_* logs/mamba_seed42_*
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict


def load_run(run_dir: Path) -> Dict:
    """Load training log and config from a run directory."""
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    with open(run_dir / "training_log.json") as f:
        log = json.load(f)
    return {"config": config, "log": log, "path": run_dir}


def smooth(values: List[float], window: int = 10) -> np.ndarray:
    """Moving average smoothing."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training_curves(runs: List[Dict], save_dir: Path):
    """Plot reward curves for all runs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {"lstm": "#2196F3", "mamba": "#FF5722"}
    labels_seen = set()

    for run in runs:
        model_type = run["config"]["model_type"]
        log = run["log"]
        color = colors.get(model_type, "#666")

        updates = [e["update"] for e in log]
        rewards = [e["mean_reward_100"] for e in log]
        policy_loss = [e["policy_loss"] for e in log]
        value_loss = [e["value_loss"] for e in log]
        entropy = [e["entropy"] for e in log]

        label = model_type.upper() if model_type not in labels_seen else None
        labels_seen.add(model_type)

        # Reward curve
        axes[0, 0].plot(updates, smooth(rewards), color=color, label=label, alpha=0.8)
        axes[0, 0].fill_between(
            updates[:len(smooth(rewards))],
            smooth(rewards) - np.std(rewards) * 0.3,
            smooth(rewards) + np.std(rewards) * 0.3,
            alpha=0.15, color=color,
        )

        # Policy loss
        axes[0, 1].plot(updates, smooth(policy_loss), color=color, alpha=0.8)

        # Value loss
        axes[1, 0].plot(updates, smooth(value_loss), color=color, alpha=0.8)

        # Entropy
        axes[1, 1].plot(updates, smooth(entropy), color=color, alpha=0.8)

    axes[0, 0].set_title("Mean Episode Reward (100-ep window)")
    axes[0, 0].set_xlabel("Update")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("Policy Loss")
    axes[0, 1].set_xlabel("Update")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("Value Loss")
    axes[1, 0].set_xlabel("Update")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("Policy Entropy")
    axes[1, 1].set_xlabel("Update")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("LSTM vs Mamba: Adaptive PID Training", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "training_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training_comparison.png")


def plot_adaptation_analysis(runs: List[Dict], save_dir: Path):
    """
    Plot post-perturbation adaptation metrics.
    Requires eval checkpoint data with detailed episode info.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {"lstm": "#2196F3", "mamba": "#FF5722"}

    # Placeholder: in practice, you'd load eval episode data from checkpoints
    # This shows the structure for when you add detailed eval logging
    model_types = []
    settling_times = []
    post_errors = []
    settling_rates = []

    for run in runs:
        mt = run["config"]["model_type"]
        # Check for eval results in the last checkpoint
        checkpoints = sorted(run["path"].glob("checkpoint_*.pt"))
        if checkpoints:
            import torch
            ckpt = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
            if "eval_results" in ckpt:
                er = ckpt["eval_results"]
                model_types.append(mt)
                settling_times.append(er.get("mean_settling_time", float("inf")))
                post_errors.append(er.get("mean_error_post", 0))
                settling_rates.append(er.get("settling_rate", 0))

    if model_types:
        x = np.arange(len(model_types))
        labels = [mt.upper() for mt in model_types]

        axes[0].bar(x, settling_times, color=[colors[mt] for mt in model_types])
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels)
        axes[0].set_title("Mean Settling Time (s)")
        axes[0].set_ylabel("Seconds")

        axes[1].bar(x, post_errors, color=[colors[mt] for mt in model_types])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels)
        axes[1].set_title("Mean Post-Perturbation Error")
        axes[1].set_ylabel("Error (m)")

        axes[2].bar(x, settling_rates, color=[colors[mt] for mt in model_types])
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(labels)
        axes[2].set_title("Settling Success Rate")
        axes[2].set_ylabel("Rate")
        axes[2].set_ylim(0, 1)

    plt.suptitle("Post-Perturbation Adaptation: LSTM vs Mamba", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "adaptation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved adaptation_comparison.png")


def print_summary(runs: List[Dict]):
    """Print a text summary table."""
    print("\n" + "=" * 70)
    print(f"{'Model':<10} {'Final Reward':>14} {'Param Count':>14} {'Best Eval':>14}")
    print("-" * 70)

    for run in runs:
        mt = run["config"]["model_type"].upper()
        log = run["log"]
        final_reward = log[-1]["mean_reward_100"] if log else 0

        # Get param count from config
        hidden = run["config"]["model"]["hidden_size"]

        # Check for best eval
        import torch
        best_path = run["path"] / "best_model.pt"
        best_eval = "N/A"
        checkpoints = sorted(run["path"].glob("checkpoint_*.pt"))
        if checkpoints:
            ckpt = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
            if "eval_results" in ckpt:
                best_eval = f"{ckpt['eval_results']['mean_reward']:.2f}"

        print(f"{mt:<10} {final_reward:>14.2f} {hidden:>14} {best_eval:>14}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Compare LSTM vs Mamba runs")
    parser.add_argument(
        "--runs", nargs="+", required=True,
        help="Paths to run directories"
    )
    parser.add_argument(
        "--output", type=str, default="comparison_results",
        help="Output directory for plots"
    )
    args = parser.parse_args()

    save_dir = Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for run_path in args.runs:
        p = Path(run_path)
        if p.exists() and (p / "training_log.json").exists():
            runs.append(load_run(p))
        else:
            print(f"Warning: {run_path} not found or incomplete, skipping")

    if not runs:
        print("No valid runs found!")
        return

    print(f"Loaded {len(runs)} runs")
    print_summary(runs)
    plot_training_curves(runs, save_dir)
    plot_adaptation_analysis(runs, save_dir)
    print(f"\nAll plots saved to {save_dir}")


if __name__ == "__main__":
    main()