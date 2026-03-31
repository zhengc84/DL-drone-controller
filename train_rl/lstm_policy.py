"""
LSTM-based Actor-Critic policy for recurrent PPO.

Processes observation sequences through an LSTM backbone and outputs
PID gain adjustments (actor) + value estimate (critic).
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple, Optional


class LSTMPolicy(nn.Module):
    """
    Recurrent actor-critic with LSTM backbone.

    Architecture:
        obs -> encoder MLP -> LSTM -> actor head -> action distribution
                                   -> critic head -> value
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 9,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        log_std_init: float = -1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Actor head (outputs mean of Gaussian)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
            nn.Tanh(),  # bound to [-1, 1]
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

    def get_initial_hidden(self, batch_size: int = 1, device: str = "cpu"):
        """Return zero-initialized LSTM hidden state."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Forward pass.

        Args:
            obs: (batch, seq_len, obs_dim) or (batch, obs_dim) observation tensor
            hidden: (h, c) LSTM hidden state
            masks: (batch, seq_len) done masks for resetting hidden states

        Returns:
            action: sampled action
            log_prob: log probability of action
            value: value estimate
            hidden: updated hidden state
        """
        # Handle single-step input
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (batch, 1, obs_dim)

        batch_size, seq_len, _ = obs.shape

        if hidden is None:
            hidden = self.get_initial_hidden(batch_size, obs.device)

        # Encode observations
        encoded = self.obs_encoder(obs)  # (batch, seq_len, hidden)

        # Reset hidden states at episode boundaries if masks provided
        if masks is not None and seq_len > 1:
            lstm_out = []
            h, c = hidden
            for t in range(seq_len):
                # Mask: 0 at done, 1 otherwise
                mask_t = masks[:, t].unsqueeze(0).unsqueeze(-1)  # (1, batch, 1)
                h = h * mask_t
                c = c * mask_t
                out, (h, c) = self.lstm(encoded[:, t:t+1, :], (h, c))
                lstm_out.append(out)
            lstm_out = torch.cat(lstm_out, dim=1)
            hidden = (h, c)
        else:
            lstm_out, hidden = self.lstm(encoded, hidden)

        # Actor
        action_mean = self.actor_mean(lstm_out)
        log_std = self.actor_log_std.clamp(-4.0, 0.0)
        action_std = log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Critic
        value = self.critic(lstm_out).squeeze(-1)

        return action, log_prob, value, hidden

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[Tuple] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions (for PPO update).

        Returns:
            log_prob, value, entropy
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)

        batch_size = obs.shape[0]
        if hidden is None:
            hidden = self.get_initial_hidden(batch_size, obs.device)

        encoded = self.obs_encoder(obs)
        lstm_out, _ = self.lstm(encoded, hidden)

        action_mean = self.actor_mean(lstm_out)
        log_std = self.actor_log_std.clamp(-4.0, 0.0)
        action_std = log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)

        if actions.dim() == 2:
            actions = actions.unsqueeze(1)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(lstm_out).squeeze(-1)

        return log_prob.squeeze(1), value.squeeze(1), entropy.squeeze(1)