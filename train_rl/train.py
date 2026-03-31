"""
Recurrent PPO training loop for adaptive PID tuning.

Supports both LSTM and Mamba policy backbones with identical training
procedures for fair comparison.

Usage:
    python train.py --model lstm --episodes 5000 --seed 42
    python train.py --model mamba --episodes 5000 --seed 42
"""

import argparse
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import deque

from config import Config, EnvConfig, ModelConfig, TrainConfig
from envs import AdaptivePIDEnv
from lstm_policy import LSTMPolicy
from mamba_policy import MambaPolicy
from pretrained_policy import PretrainedRLPolicy
from utils import RecurrentRolloutBuffer, RewardNormalizer


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_policy(model_type: str, obs_dim: int, cfg: Config,
                 detector_weights: str = None, detector_type: str = "mamba") -> nn.Module:
    """Instantiate the correct policy based on model_type."""
    mc = cfg.model

    if model_type == "lstm":
        return LSTMPolicy(
            obs_dim=obs_dim,
            action_dim=mc.action_dim,
            hidden_size=mc.hidden_size,
            num_layers=mc.num_layers,
            dropout=mc.dropout,
            log_std_init=mc.log_std_init,
        )
    elif model_type == "mamba":
        return MambaPolicy(
            obs_dim=obs_dim,
            action_dim=mc.action_dim,
            hidden_size=mc.hidden_size,
            num_layers=mc.num_layers,
            d_state=mc.mamba_d_state,
            d_conv=mc.mamba_d_conv,
            expand=mc.mamba_expand,
            log_std_init=mc.log_std_init,
        )
    elif model_type == "pretrained":
        return PretrainedRLPolicy(
            obs_dim=obs_dim,
            action_dim=mc.action_dim,
            detector_type=detector_type,
            detector_weights=detector_weights,
            hidden_size=mc.hidden_size,
            error_history_len=cfg.env.error_history_len,
            log_std_init=mc.log_std_init,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def ppo_update(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: RecurrentRolloutBuffer,
    cfg: TrainConfig,
) -> dict:
    """Run PPO update epochs on the collected rollout data."""
    tc = cfg
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    for epoch in range(tc.num_epochs):
        for batch in buffer.get_minibatches(tc.minibatch_size):
            obs = batch["obs"]          # (B, seq_len, obs_dim)
            actions = batch["actions"]  # (B, seq_len, action_dim)
            old_lp = batch["old_log_probs"]  # (B, seq_len)
            advantages = batch["advantages"]  # (B, seq_len)
            returns = batch["returns"]  # (B, seq_len)
            masks = batch["masks"]      # (B, seq_len)

            # Evaluate actions across the full sequence
            # We process each timestep and collect results
            B, S, _ = obs.shape
            all_log_probs = []
            all_values = []
            all_entropy = []

            for t in range(S):
                lp, v, ent = policy.evaluate_actions(
                    obs[:, t, :], actions[:, t, :]
                )
                all_log_probs.append(lp)
                all_values.append(v)
                all_entropy.append(ent)

            new_log_probs = torch.stack(all_log_probs, dim=1)  # (B, S)
            values = torch.stack(all_values, dim=1)  # (B, S)
            entropy = torch.stack(all_entropy, dim=1)  # (B, S)

            # Apply masks (ignore padded timesteps)
            valid = masks.bool()

            # PPO clipped loss
            ratio = (new_log_probs - old_lp).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - tc.clip_epsilon, 1 + tc.clip_epsilon) * advantages

            policy_loss = -torch.min(surr1, surr2)[valid].mean()
            value_loss = ((values - returns) ** 2)[valid].mean()
            entropy_loss = -entropy[valid].mean()

            loss = (
                policy_loss
                + tc.value_coef * value_loss
                + tc.entropy_coef * entropy_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), tc.max_grad_norm)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += (-entropy_loss).item()
            n_updates += 1

    return {
        "policy_loss": total_policy_loss / max(n_updates, 1),
        "value_loss": total_value_loss / max(n_updates, 1),
        "entropy": total_entropy / max(n_updates, 1),
    }


def evaluate(
    policy: nn.Module,
    cfg: Config,
    n_episodes: int = 10,
    device: str = "cpu",
) -> dict:
    """Run evaluation episodes and return metrics."""
    env = AdaptivePIDEnv(config=cfg.env)
    policy.eval()

    rewards_list = []
    errors_pre_perturb = []
    errors_post_perturb = []
    settling_times = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        hidden = policy.get_initial_hidden(1, device)
        done = False
        ep_reward = 0.0
        pre_errors = []
        post_errors = []
        settled = False
        settle_step = None

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, hidden = policy(obs_t, hidden)
            action_np = action.squeeze(0).cpu().numpy()
            if action_np.ndim > 1:
                action_np = action_np.squeeze(0)

            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            ep_reward += reward

            err = info["error_norm"]
            if info["mass_perturbed"]:
                post_errors.append(err)
                if not settled and err < cfg.env.settling_tolerance:
                    settled = True
                    settle_step = env.step_count
            else:
                pre_errors.append(err)

        rewards_list.append(ep_reward)
        if pre_errors:
            errors_pre_perturb.append(np.mean(pre_errors))
        if post_errors:
            errors_post_perturb.append(np.mean(post_errors))
        if settle_step is not None:
            settling_times.append(settle_step * env.ctrl_dt)

    env.close()
    policy.train()

    return {
        "mean_reward": np.mean(rewards_list),
        "std_reward": np.std(rewards_list),
        "mean_error_pre": np.mean(errors_pre_perturb) if errors_pre_perturb else 0,
        "mean_error_post": np.mean(errors_post_perturb) if errors_post_perturb else 0,
        "mean_settling_time": np.mean(settling_times) if settling_times else float("inf"),
        "settling_rate": len(settling_times) / n_episodes,
    }


def collect_rollout(
    env: AdaptivePIDEnv,
    policy: nn.Module,
    buffer: RecurrentRolloutBuffer,
    n_steps: int,
    device: str = "cpu",
    reward_normalizer: RewardNormalizer = None,
) -> dict:
    """Collect n_steps of experience into the buffer."""
    obs, info = env.reset()
    hidden = policy.get_initial_hidden(1, device)
    ep_rewards = []
    current_ep_reward = 0.0

    for step_i in range(n_steps):
        if step_i % 500 == 0:
            print(f"  Collecting step {step_i}/{n_steps}...", end="\r")

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, value, hidden = policy(obs_t, hidden)

        action_np = action.squeeze(0).cpu().numpy()
        if action_np.ndim > 1:
            action_np = action_np.squeeze(0)
        lp = log_prob.item()
        val = value.item()

        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        current_ep_reward += reward

        # Normalize reward for the buffer (raw reward tracked separately)
        if reward_normalizer is not None:
            norm_reward = reward_normalizer.normalize(reward, done)
        else:
            norm_reward = reward

        buffer.add(obs, action_np, norm_reward, val, lp, done)

        if done:
            ep_rewards.append(current_ep_reward)
            current_ep_reward = 0.0
            obs, info = env.reset()
            hidden = policy.get_initial_hidden(1, device)
        else:
            obs = next_obs

    # Bootstrap last value
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        _, _, last_val, _ = policy(obs_t, hidden)
    buffer.compute_gae(last_val.item())

    return {
        "ep_rewards": ep_rewards,
        "mean_ep_reward": np.mean(ep_rewards) if ep_rewards else 0.0,
    }


def train(model_type: str, cfg: Config, detector_weights: str = None,
          detector_type: str = "mamba", unfreeze_at: int = None):
    """Main training function."""
    tc = cfg.train
    device = tc.device
    set_seed(tc.seed)

    # Setup logging
    run_name = f"{model_type}_seed{tc.seed}_{int(time.time())}"
    log_dir = Path(tc.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(log_dir / "config.json", "w") as f:
        json.dump({
            "model_type": model_type,
            "env": vars(cfg.env),
            "model": vars(cfg.model),
            "train": {k: v for k, v in vars(tc).items() if k != "device"},
            "detector_weights": detector_weights,
            "detector_type": detector_type,
            "unfreeze_at": unfreeze_at,
        }, f, indent=2, default=str)

    # Create env and policy
    env = AdaptivePIDEnv(config=cfg.env)
    obs_dim = env.observation_space.shape[0]

    policy = build_policy(
        model_type, obs_dim, cfg,
        detector_weights=detector_weights,
        detector_type=detector_type,
    ).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=tc.learning_rate,
    )

    param_total = sum(p.numel() for p in policy.parameters())
    param_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} policy")
    print(f"Parameters: {param_trainable:,} trainable / {param_total:,} total")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {cfg.model.action_dim}")
    print(f"LR: {tc.learning_rate} | Value coef: {tc.value_coef} | Entropy coef: {tc.entropy_coef}")
    print(f"Reward normalization: {tc.normalize_rewards}")
    if model_type == "pretrained":
        print(f"Detector: {detector_type} from {detector_weights}")
        print(f"Unfreeze at: update {unfreeze_at}" if unfreeze_at else "Detector stays frozen")
    print(f"Device: {device}")
    print(f"Log dir: {log_dir}")
    print(f"{'='*60}\n")

    # Training loop
    buffer = RecurrentRolloutBuffer(
        buffer_size=tc.steps_per_update,
        obs_dim=obs_dim,
        action_dim=cfg.model.action_dim,
        seq_len=tc.seq_len,
        gamma=tc.gamma,
        gae_lambda=tc.gae_lambda,
        device=device,
    )

    reward_history = deque(maxlen=100)
    best_eval_reward = -float("inf")
    training_log = []

    # Reward normalization
    reward_norm = RewardNormalizer(gamma=tc.gamma, clip=tc.reward_clip) if tc.normalize_rewards else None

    total_updates = tc.total_episodes  # repurpose as total update iterations
    for update in range(1, total_updates + 1):
        t_start = time.time()

        # Collect rollout
        buffer.reset()
        rollout_info = collect_rollout(
            env, policy, buffer, tc.steps_per_update, device,
            reward_normalizer=reward_norm,
        )

        for r in rollout_info["ep_rewards"]:
            reward_history.append(r)

        # PPO update
        update_info = ppo_update(policy, optimizer, buffer, tc)

        # Unfreeze pretrained detector if scheduled
        if (unfreeze_at is not None and update == unfreeze_at
                and model_type == "pretrained"):
            policy.unfreeze_detector()
            # Rebuild optimizer to include newly unfrozen params with lower LR
            optimizer = torch.optim.Adam([
                {"params": policy.detector.parameters(), "lr": tc.learning_rate * 0.1},
                {"params": policy.actor.parameters()},
                {"params": [policy.actor_log_std]},
                {"params": policy.critic.parameters()},
            ], lr=tc.learning_rate)

        elapsed = time.time() - t_start

        # Logging
        if update % tc.log_interval == 0:
            mean_reward = np.mean(reward_history) if reward_history else 0
            log_entry = {
                "update": update,
                "mean_reward_100": float(mean_reward),
                "rollout_mean": float(rollout_info["mean_ep_reward"]),
                "policy_loss": update_info["policy_loss"],
                "value_loss": update_info["value_loss"],
                "entropy": update_info["entropy"],
                "time_per_update": elapsed,
            }
            training_log.append(log_entry)

            print(
                f"[Update {update:5d}] "
                f"Reward: {mean_reward:8.2f} | "
                f"PL: {update_info['policy_loss']:7.4f} | "
                f"VL: {update_info['value_loss']:7.4f} | "
                f"Ent: {update_info['entropy']:6.3f} | "
                f"Time: {elapsed:.1f}s"
            )

        # Evaluation
        if update % tc.save_interval == 0:
            eval_results = evaluate(policy, cfg, tc.eval_episodes, device)
            print(
                f"\n  EVAL | Reward: {eval_results['mean_reward']:.2f} "
                f"| Pre-err: {eval_results['mean_error_pre']:.4f} "
                f"| Post-err: {eval_results['mean_error_post']:.4f} "
                f"| Settle: {eval_results['mean_settling_time']:.2f}s "
                f"| Rate: {eval_results['settling_rate']:.0%}\n"
            )

            # Save best model
            if eval_results["mean_reward"] > best_eval_reward:
                best_eval_reward = eval_results["mean_reward"]
                torch.save(policy.state_dict(), log_dir / "best_model.pt")

            # Save checkpoint
            torch.save({
                "update": update,
                "model_state": policy.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "eval_results": eval_results,
            }, log_dir / f"checkpoint_{update}.pt")

    # Save final model and log
    torch.save(policy.state_dict(), log_dir / "final_model.pt")
    with open(log_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    env.close()
    print(f"\nTraining complete. Results saved to {log_dir}")
    return log_dir


def main():
    parser = argparse.ArgumentParser(description="Train adaptive PID with RL")
    parser.add_argument(
        "--model", type=str, required=True, choices=["lstm", "mamba", "pretrained"],
        help="Sequence model backbone"
    )
    parser.add_argument("--episodes", type=int, default=None, help="Total training updates")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=None, help="Hidden size")
    parser.add_argument("--target", type=str, default=None, help="Target type: hover/step/sinusoid")
    parser.add_argument("--entropy", type=float, default=None, help="Entropy coefficient")
    parser.add_argument("--render", action="store_true", help="Render PyBullet GUI")

    # Pretrained detector args
    parser.add_argument("--detector-weights", type=str, default=None,
                       help="Path to Phase 1 weights (e.g. Mamba_Tuner_weights.pth)")
    parser.add_argument("--detector-type", type=str, default="mamba",
                       choices=["lstm", "mamba"],
                       help="Phase 1 model architecture to load")
    parser.add_argument("--unfreeze-at", type=int, default=None,
                       help="Update step to unfreeze pretrained detector for fine-tuning")
    args = parser.parse_args()

    cfg = Config()

    if args.episodes is not None:
        cfg.train.total_episodes = args.episodes
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.lr is not None:
        cfg.train.learning_rate = args.lr
    if args.hidden is not None:
        cfg.model.hidden_size = args.hidden
    if args.target is not None:
        cfg.env.target_type = args.target
    if args.entropy is not None:
        cfg.train.entropy_coef = args.entropy

    if args.model == "pretrained" and args.detector_weights is None:
        parser.error("--detector-weights is required when using --model pretrained")

    train(
        args.model, cfg,
        detector_weights=args.detector_weights,
        detector_type=args.detector_type,
        unfreeze_at=args.unfreeze_at,
    )


if __name__ == "__main__":
    main()