"""
Load all checkpoints from LSTM and Mamba runs and compare eval results.

Usage:
    python analyze_checkpoints.py
    python analyze_checkpoints.py --log-dir logs
"""

import torch
import glob
import json
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_all_checkpoints(run_dir):
    """Load eval results from all checkpoints in a run directory."""
    run_dir = Path(run_dir)
    checkpoints = sorted(run_dir.glob("checkpoint_*.pt"),
                         key=lambda p: int(p.stem.split("_")[1]))

    results = []
    for ckpt_path in checkpoints:
        # Extract update number from filename
        update = int(ckpt_path.stem.split("_")[1])
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if "eval_results" in ckpt:
            entry = {"update": update}
            entry.update(ckpt["eval_results"])
            results.append(entry)

    results.sort(key=lambda x: x["update"])
    return results


def load_training_log(run_dir):
    """Load the training log JSON."""
    log_path = Path(run_dir) / "training_log.json"
    if log_path.exists():
        with open(log_path) as f:
            return json.load(f)
    return []


def print_comparison(runs):
    """Print a formatted comparison table."""
    for name, data in runs.items():
        print(f"\n{'='*70}")
        print(f"  {name.upper()}")
        print(f"{'='*70}")
        print(f"{'Update':>8} | {'Reward':>10} | {'Pre-err':>10} | {'Post-err':>10} | "
              f"{'Settle':>8} | {'Rate':>6}")
        print(f"{'-'*70}")

        for entry in data:
            print(
                f"{entry['update']:>8} | "
                f"{entry['mean_reward']:>10.2f} | "
                f"{entry['mean_error_pre']:>10.4f} | "
                f"{entry['mean_error_post']:>10.4f} | "
                f"{entry['mean_settling_time']:>7.2f}s | "
                f"{entry['settling_rate']:>5.0%}"
            )

    # Side-by-side final comparison
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON (last checkpoint)")
    print(f"{'='*70}")
    print(f"{'Metric':<25}", end="")
    for name in runs:
        print(f"| {name.upper():>15} ", end="")
    print()
    print("-" * (25 + 18 * len(runs)))

    metrics = [
        ("Reward", "mean_reward", ".2f"),
        ("Pre-err (m)", "mean_error_pre", ".4f"),
        ("Post-err (m)", "mean_error_post", ".4f"),
        ("Settling time (s)", "mean_settling_time", ".2f"),
        ("Settling rate", "settling_rate", ".0%"),
    ]

    for label, key, fmt in metrics:
        print(f"{label:<25}", end="")
        values = []
        for name, data in runs.items():
            val = data[-1][key]
            values.append(val)
            print(f"| {val:>15{fmt}} ", end="")

        # Highlight winner
        if key == "settling_rate":
            best = max(values)
        elif key == "mean_reward":
            best = max(values)
        else:
            best = min(values)

        winner_idx = values.index(best)
        winner_name = list(runs.keys())[winner_idx]
        print(f"  ← {winner_name}")

    print()


def plot_eval_curves(runs, save_dir):
    """Plot eval metrics over training for all runs."""
    colors = {
        "lstm": "#2196F3",
        "mamba": "#FF5722",
        "pretrained_lstm": "#4CAF50",
        "pretrained_mamba": "#9C27B0",
    }
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics = [
        ("mean_reward", "Eval Reward", axes[0, 0]),
        ("mean_error_pre", "Pre-perturbation Error (m)", axes[0, 1]),
        ("mean_error_post", "Post-perturbation Error (m)", axes[0, 2]),
        ("mean_settling_time", "Settling Time (s)", axes[1, 0]),
        ("settling_rate", "Settling Rate", axes[1, 1]),
        ("std_reward", "Reward Std Dev", axes[1, 2]),
    ]

    for name, data in runs.items():
        updates = [e["update"] for e in data]
        color = colors.get(name, "#666")

        for key, title, ax in metrics:
            values = [e[key] for e in data]
            ax.plot(updates, values, marker="o", markersize=4,
                    color=color, label=name.upper(), linewidth=2)
            ax.set_title(title)
            ax.set_xlabel("Update")
            ax.grid(True, alpha=0.3)
            ax.legend()

    plt.suptitle("LSTM vs Mamba: Eval Metrics Over Training", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = Path(save_dir) / "eval_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved eval comparison plot to {save_path}")


def plot_training_curves(runs_logs, save_dir):
    """Plot training metrics (reward, loss, entropy) from training logs."""
    colors = {
        "lstm": "#2196F3",
        "mamba": "#FF5722",
        "pretrained_lstm": "#4CAF50",
        "pretrained_mamba": "#9C27B0",
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for name, log in runs_logs.items():
        if not log:
            continue

        updates = [e["update"] for e in log]
        color = colors.get(name, "#666")

        # Smooth with moving average
        def smooth(vals, window=10):
            if len(vals) < window:
                return vals
            kernel = np.ones(window) / window
            return np.convolve(vals, kernel, mode="valid")

        rewards = [e["mean_reward_100"] for e in log]
        pl = [e["policy_loss"] for e in log]
        vl = [e["value_loss"] for e in log]
        ent = [e["entropy"] for e in log]

        axes[0, 0].plot(updates[:len(smooth(rewards))], smooth(rewards),
                       color=color, label=name.upper(), linewidth=1.5, alpha=0.8)
        axes[0, 1].plot(updates[:len(smooth(pl))], smooth(pl),
                       color=color, linewidth=1.5, alpha=0.8)
        axes[1, 0].plot(updates[:len(smooth(vl))], smooth(vl),
                       color=color, linewidth=1.5, alpha=0.8)
        axes[1, 1].plot(updates[:len(smooth(ent))], smooth(ent),
                       color=color, linewidth=1.5, alpha=0.8)

    axes[0, 0].set_title("Training Reward (100-ep avg)")
    axes[0, 0].set_xlabel("Update")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("Policy Loss")
    axes[0, 1].set_xlabel("Update")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("Value Loss")
    axes[1, 0].set_xlabel("Update")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("Entropy")
    axes[1, 1].set_xlabel("Update")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("LSTM vs Mamba: Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = Path(save_dir) / "training_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training comparison plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze all checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs", help="Base log directory")
    parser.add_argument("--output", type=str, default="analysis_results", help="Output directory")
    args = parser.parse_args()

    # Find all run directories
    run_dirs = sorted(glob.glob(f"{args.log_dir}/*"))
    if not run_dirs:
        print(f"No runs found in {args.log_dir}/")
        return

    print(f"Found {len(run_dirs)} runs:")
    for d in run_dirs:
        print(f"  {d}")

    # Group by model type
    runs_eval = {}
    runs_logs = {}
    for d in run_dirs:
        config_path = Path(d) / "config.json"
        if not config_path.exists():
            continue

        with open(config_path) as f:
            config = json.load(f)

        model_type = config.get("model_type", Path(d).name.split("_")[0])
        # Distinguish pretrained runs by detector type
        if model_type == "pretrained":
            det_type = config.get("detector_type", "unknown")
            model_type = f"pretrained_{det_type}"

        checkpoints = load_all_checkpoints(d)
        if checkpoints:
            runs_eval[model_type] = checkpoints

        training_log = load_training_log(d)
        if training_log:
            runs_logs[model_type] = training_log

    if not runs_eval:
        print("No checkpoint data found!")
        return

    # Print comparison
    print_comparison(runs_eval)

    # Generate plots
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_eval_curves(runs_eval, output_dir)
    plot_training_curves(runs_logs, output_dir)

    # Save raw data as JSON for further analysis
    raw_data = {}
    for name, data in runs_eval.items():
        raw_data[name] = []
        for entry in data:
            clean = {k: float(v) if isinstance(v, (np.floating, float)) else v
                     for k, v in entry.items()}
            raw_data[name].append(clean)

    with open(output_dir / "eval_data.json", "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"Saved raw eval data to {output_dir}/eval_data.json")


if __name__ == "__main__":
    main()