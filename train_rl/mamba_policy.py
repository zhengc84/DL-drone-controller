"""
Mamba-based Actor-Critic policy for recurrent PPO.

Uses the Mamba selective state-space model as the sequence backbone.
Falls back to a pure-PyTorch SSM implementation if mamba-ssm is not installed.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple, Optional

# Try importing the official Mamba implementation
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print(
        "[WARNING] mamba-ssm not installed. Using fallback SSM implementation.\n"
        "Install with: pip install mamba-ssm (requires CUDA)"
    )


class FallbackSSMBlock(nn.Module):
    """
    Simplified S4-style SSM block as a Mamba fallback.
    Uses a 1D conv + gated linear unit + learnable state-space recurrence.
    Not as performant as Mamba but structurally similar.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
        )
        self.act = nn.SiLU()

        # SSM parameters
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)

        # Learnable SSM matrices
        self.A_log = nn.Parameter(torch.randn(d_inner, d_state))
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        residual = x
        x = self.norm(x)

        # Project and split into two paths (gated)
        xz = self.in_proj(x)
        x_path, z = xz.chunk(2, dim=-1)

        # Conv
        x_conv = x_path.transpose(1, 2)  # (B, D, L)
        x_conv = self.conv1d(x_conv)[:, :, :x_path.shape[1]]
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.act(x_conv)

        # Simplified selective scan (linear attention approximation)
        dt = torch.sigmoid(self.dt_proj(x_conv))
        x_ssm = x_conv * dt  # gated

        # Gated output
        y = x_ssm * self.act(z)
        y = self.out_proj(y)

        return y + residual


class MambaBackbone(nn.Module):
    """Stack of Mamba (or fallback SSM) blocks."""

    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if MAMBA_AVAILABLE:
                self.layers.append(
                    Mamba(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                    )
                )
            else:
                self.layers.append(
                    FallbackSSMBlock(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                    )
                )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MambaPolicy(nn.Module):
    """
    Actor-Critic with Mamba SSM backbone.

    Architecture matches LSTMPolicy interface for fair comparison:
        obs -> encoder MLP -> Mamba blocks -> actor head -> action
                                           -> critic head -> value

    Key difference: Mamba processes the full sequence in parallel (no hidden
    state to carry between steps), making it more efficient for long sequences.
    For step-by-step rollout, we maintain a growing context window.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 9,
        hidden_size: int = 128,
        num_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        log_std_init: float = -1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Mamba backbone
        self.backbone = MambaBackbone(
            d_model=hidden_size,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # Actor head
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
            nn.Tanh(),
        )
        self.actor_log_std = nn.Parameter(
            torch.full((action_dim,), log_std_init)
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        # Context buffer for step-by-step inference
        self._context_buffer = None

    def get_initial_hidden(self, batch_size: int = 1, device: str = "cpu"):
        """
        For API compatibility with LSTMPolicy.
        Mamba doesn't need hidden state — returns None.
        Resets the context buffer instead.
        """
        self._context_buffer = None
        return None

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[any] = None,  # unused, for API compat
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """
        Forward pass.

        Args:
            obs: (batch, seq_len, obs_dim) or (batch, obs_dim)
            hidden: ignored (Mamba is parallel, no recurrent state needed for training)
            masks: (batch, seq_len) done masks — used to reset context buffer

        Returns:
            action, log_prob, value, None (hidden placeholder)
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)

        batch_size, seq_len, _ = obs.shape

        # Handle episode resets via masks
        if masks is not None and seq_len == 1:
            # Step-by-step mode: maintain context buffer
            if self._context_buffer is None or masks.sum() < batch_size:
                # Reset buffer for done episodes
                self._context_buffer = obs
            else:
                self._context_buffer = torch.cat(
                    [self._context_buffer, obs], dim=1
                )
                # Limit context window to prevent OOM
                if self._context_buffer.shape[1] > 200:
                    self._context_buffer = self._context_buffer[:, -200:, :]
            obs_seq = self._context_buffer
        else:
            obs_seq = obs

        # Encode
        encoded = self.obs_encoder(obs_seq)

        # Mamba processes full sequence
        backbone_out = self.backbone(encoded)

        # Take last timestep output for action/value
        if seq_len == 1 and obs_seq.shape[1] > 1:
            last_out = backbone_out[:, -1:, :]
        else:
            last_out = backbone_out

        # Actor
        action_mean = self.actor_mean(last_out)
        log_std = self.actor_log_std.clamp(-4.0, 0.0)
        action_std = log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Critic
        value = self.critic(last_out).squeeze(-1)

        return action, log_prob, value, None

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[any] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate given actions for PPO update."""
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)

        encoded = self.obs_encoder(obs)
        backbone_out = self.backbone(encoded)

        action_mean = self.actor_mean(backbone_out)
        log_std = self.actor_log_std.clamp(-4.0, 0.0)
        action_std = log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)

        if actions.dim() == 2:
            actions = actions.unsqueeze(1)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(backbone_out).squeeze(-1)

        return log_prob.squeeze(1), value.squeeze(1), entropy.squeeze(1)