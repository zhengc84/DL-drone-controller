"""
Bridge between Phase 1 (supervised, 1DOF) and Phase 2 (RL, 6DOF).

Loads the pretrained Mamba or LSTM backbone from Phase 1 and reuses it
as a per-axis "plant change detector" inside the RL policy.

Phase 1 architecture (from your code):
    input (B, 50, 4) → Mamba/LSTM → last step → Linear → 3 gains
    
    The backbone learned: "given 50 steps of (error, derror, ierror, ?),
    extract a representation that predicts what gains should be."

Phase 2 reuse:
    For each axis (x, y, z), feed that axis's error history through
    the SAME pretrained backbone. Concatenate the three embeddings.
    RL policy head sits on top and outputs 9 gain deltas.

This works because mass changes look the same in the error signal
regardless of axis. The pretrained model already knows how to detect
"the plant got heavier/lighter" from error trajectories.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple, Optional

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class PretrainedDetector(nn.Module):
    """
    Wraps a Phase 1 model's backbone (everything except the final output layer)
    as a frozen feature extractor.
    """

    def __init__(self, model_type: str = "mamba", weights_path: str = None):
        super().__init__()
        self.model_type = model_type

        if model_type == "mamba":
            # Match Phase 1 architecture exactly
            self.input_proj = nn.Linear(4, 256)
            self.backbone = Mamba(
                d_model=256,
                d_state=16,
                d_conv=4,
                expand=2,
            )
            self.embed_dim = 256
        elif model_type == "lstm":
            self.backbone = nn.LSTM(
                input_size=4,
                hidden_size=64,
                num_layers=2,
                batch_first=True,
            )
            self.embed_dim = 64
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load Phase 1 weights (minus the output head)
        if weights_path is not None:
            self._load_phase1_weights(weights_path)

        # Freeze the backbone — RL trains the head on top
        self.freeze()

    def _load_phase1_weights(self, weights_path: str):
        """Load Phase 1 weights, skipping the final output projection."""
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

        # Filter out the output head weights
        # Phase 1 Mamba: keys are input_proj.*, mamba.*, output_proj.*
        # Phase 1 LSTM: keys are lstm.*, fc.*
        if self.model_type == "mamba":
            skip_prefixes = ["output_proj"]
        else:
            skip_prefixes = ["fc"]

        filtered = {
            k: v for k, v in state_dict.items()
            if not any(k.startswith(p) for p in skip_prefixes)
        }

        # Load matching weights
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        print(f"[PretrainedDetector] Loaded {len(filtered)} tensors from {weights_path}")
        if missing:
            print(f"  Missing (expected): {missing}")
        if unexpected:
            print(f"  Unexpected (skipped): {unexpected}")

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 4) — single-axis error features
               [error, d_error/dt, integral_error, time_or_flag]

        Returns:
            embedding: (batch, embed_dim)
        """
        if self.model_type == "mamba":
            x = self.input_proj(x)
            x = self.backbone(x)
            return x[:, -1, :]  # last timestep
        else:
            lstm_out, _ = self.backbone(x)
            return lstm_out[:, -1, :]


class PretrainedRLPolicy(nn.Module):
    """
    RL policy that uses a pretrained Phase 1 detector per axis.

    Architecture:
        For each axis (x, y, z):
            extract 4 features from obs → (error, derror, ierror, flag)
            run through shared pretrained detector → embedding

        Concatenate [x_embed, y_embed, z_embed, drone_state, current_gains]
            → MLP actor head → 9 gain deltas
            → MLP critic head → value

    The pretrained detector starts frozen. After initial RL training,
    you can unfreeze it for end-to-end fine-tuning.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 9,
        detector_type: str = "mamba",
        detector_weights: str = None,
        hidden_size: int = 128,
        error_history_len: int = 50,
        log_std_init: float = -1.0,
    ):
        super().__init__()
        self.error_history_len = error_history_len
        self.hidden_size = hidden_size

        # Pretrained detector (shared across axes)
        self.detector = PretrainedDetector(
            model_type=detector_type,
            weights_path=detector_weights,
        )
        det_dim = self.detector.embed_dim

        # After detector: 3 axis embeddings + drone state + PID gains + extras
        # drone state = 12 (pos, vel, rpy, ang_vel)
        # pid gains = 9
        # extras = 2 (perturb flag, time)
        context_dim = det_dim * 3 + 12 + 3 + 2

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(context_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh(),
        )
        self.actor_log_std = nn.Parameter(
            torch.full((action_dim,), log_std_init)
        )

        # Critic head (separate network)
        self.critic = nn.Sequential(
            nn.Linear(context_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def _extract_axis_features(self, obs: torch.Tensor):
        """
        Extract per-axis error features from the flat observation.

        The observation layout (from adaptive_pid_env.py):
            [0 : history_len*3]  — error history (x,y,z interleaved per step)
            [history_len*3 : +12] — drone state (pos, vel, rpy, ang_vel)
            [+12 : +9]           — current PID gains
            [+9 : +1]            — perturbation flag
            [+1 : +1]            — normalized time

        We reshape error history into (batch, history_len, 3) and build
        per-axis features: [error, d_error, cumulative_error, perturb_flag]
        """
        B = obs.shape[0]
        h_len = self.error_history_len
        h_end = h_len * 3

        # Error history: (B, h_len, 3) for xyz
        error_hist = obs[:, :h_end].reshape(B, h_len, 3)

        # Compute derivative (finite difference)
        d_error = torch.zeros_like(error_hist)
        d_error[:, 1:, :] = error_hist[:, 1:, :] - error_hist[:, :-1, :]

        # Compute integral (cumulative sum)
        i_error = torch.cumsum(error_hist, dim=1)

        # Perturbation flag (broadcast to all timesteps)
        perturb_flag = obs[:, -2].unsqueeze(1).unsqueeze(2).expand(B, h_len, 1)

        # Build per-axis feature tensors: (B, h_len, 4) each
        axis_features = []
        for ax in range(3):
            feat = torch.stack([
                error_hist[:, :, ax],    # error
                d_error[:, :, ax],       # d_error/dt
                i_error[:, :, ax],       # integral error
                perturb_flag[:, :, 0],   # flag
            ], dim=-1)  # (B, h_len, 4)
            axis_features.append(feat)

        # Drone state and other context
        state_start = h_end
        drone_state = obs[:, state_start:state_start + 12]
        pid_gains = obs[:, state_start + 12:state_start + 15]
        extras = obs[:, -2:]  # perturb flag + time

        return axis_features, drone_state, pid_gains, extras

    def _get_context(self, obs: torch.Tensor) -> torch.Tensor:
        """Run detector on each axis and build context vector."""
        axis_features, drone_state, pid_gains, extras = self._extract_axis_features(obs)

        # Run pretrained detector on each axis
        embeddings = []
        for ax_feat in axis_features:
            emb = self.detector(ax_feat)  # (B, det_dim)
            embeddings.append(emb)

        # Concatenate everything
        context = torch.cat(embeddings + [drone_state, pid_gains, extras], dim=-1)
        return context

    def get_initial_hidden(self, batch_size: int = 1, device: str = "cpu"):
        """API compatibility — this policy is feedforward, no hidden state."""
        return None

    def forward(
        self,
        obs: torch.Tensor,
        hidden=None,
        masks=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        if obs.dim() == 3:
            # (batch, seq_len, obs_dim) — take last timestep
            obs = obs[:, -1, :]

        context = self._get_context(obs)

        action_mean = self.actor(context)
        log_std = self.actor_log_std.clamp(-4.0, 0.0)
        action_std = log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(context).squeeze(-1)

        return action, log_prob, value, None

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        hidden=None,
        masks=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs.dim() == 3:
            obs = obs[:, -1, :]

        context = self._get_context(obs)

        action_mean = self.actor(context)
        log_std = self.actor_log_std.clamp(-4.0, 0.0)
        action_std = log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(context).squeeze(-1)

        return log_prob, value, entropy

    def unfreeze_detector(self):
        """Call after initial RL training to fine-tune the detector."""
        self.detector.unfreeze()
        print("[PretrainedRLPolicy] Detector unfrozen for fine-tuning")


# ── Usage example ──────────────────────────────────────────────────
#
# # In train.py, add "pretrained" as a model choice:
#
# from models.pretrained_policy import PretrainedRLPolicy
#
# if model_type == "pretrained":
#     policy = PretrainedRLPolicy(
#         obs_dim=obs_dim,
#         detector_type="mamba",  # or "lstm" to use your Phase 1 LSTM
#         detector_weights="path/to/Mamba_Tuner_weights.pth",
#         hidden_size=128,
#         error_history_len=cfg.env.error_history_len,
#     )
#
# # Train normally with RL for N updates with detector frozen.
# # Then unfreeze and fine-tune:
#
# if update == 200:  # after initial convergence
#     policy.unfreeze_detector()
#     # Optionally lower LR:
#     for pg in optimizer.param_groups:
#         pg['lr'] *= 0.1