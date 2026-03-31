"""Rollout buffer for recurrent PPO, handling sequence chunking."""

import torch
import numpy as np
from typing import Generator, Optional


class RecurrentRolloutBuffer:
    """
    Stores rollout data and generates sequence-chunked minibatches
    for recurrent policy training (LSTM or Mamba).
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
        seq_len: int = 50,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        self.obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = float(done)
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def compute_gae(self, last_value: float):
        """Compute Generalized Advantage Estimation."""
        size = self.ptr
        last_gae = 0.0
        for t in reversed(range(size)):
            if t == size - 1:
                next_value = last_value
                next_done = 0.0
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * (1 - next_done)
                - self.values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * last_gae
            self.advantages[t] = last_gae

        self.returns[:size] = self.advantages[:size] + self.values[:size]

    def get_minibatches(
        self, minibatch_size: int
    ) -> Generator:
        """
        Yield sequence-chunked minibatches.

        Each minibatch contains sequences of length seq_len for
        proper recurrent training (BPTT for LSTM, full context for Mamba).
        """
        size = self.ptr

        # Build sequence chunks
        # Find episode boundaries
        done_indices = np.where(self.dones[:size] == 1.0)[0]
        ep_starts = np.concatenate([[0], done_indices + 1])
        ep_starts = ep_starts[ep_starts < size]

        sequences_obs = []
        sequences_actions = []
        sequences_log_probs = []
        sequences_advantages = []
        sequences_returns = []
        sequences_masks = []

        for start in ep_starts:
            # Find episode end
            ep_dones = np.where(self.dones[start:size] == 1.0)[0]
            if len(ep_dones) > 0:
                end = start + ep_dones[0] + 1
            else:
                end = size

            ep_len = end - start

            # Chunk episode into sequences of seq_len
            for chunk_start in range(0, ep_len, self.seq_len):
                chunk_end = min(chunk_start + self.seq_len, ep_len)
                actual_len = chunk_end - chunk_start

                # Pad if needed
                obs_chunk = np.zeros((self.seq_len, self.obs_dim), dtype=np.float32)
                act_chunk = np.zeros((self.seq_len, self.action_dim), dtype=np.float32)
                lp_chunk = np.zeros(self.seq_len, dtype=np.float32)
                adv_chunk = np.zeros(self.seq_len, dtype=np.float32)
                ret_chunk = np.zeros(self.seq_len, dtype=np.float32)
                mask_chunk = np.zeros(self.seq_len, dtype=np.float32)

                idx = start + chunk_start
                obs_chunk[:actual_len] = self.obs[idx:idx + actual_len]
                act_chunk[:actual_len] = self.actions[idx:idx + actual_len]
                lp_chunk[:actual_len] = self.log_probs[idx:idx + actual_len]
                adv_chunk[:actual_len] = self.advantages[idx:idx + actual_len]
                ret_chunk[:actual_len] = self.returns[idx:idx + actual_len]
                mask_chunk[:actual_len] = 1.0

                sequences_obs.append(obs_chunk)
                sequences_actions.append(act_chunk)
                sequences_log_probs.append(lp_chunk)
                sequences_advantages.append(adv_chunk)
                sequences_returns.append(ret_chunk)
                sequences_masks.append(mask_chunk)

        n_seqs = len(sequences_obs)
        if n_seqs == 0:
            return

        # Shuffle and batch
        indices = np.random.permutation(n_seqs)
        for batch_start in range(0, n_seqs, minibatch_size):
            batch_idx = indices[batch_start:batch_start + minibatch_size]
            batch_obs = torch.tensor(
                np.array([sequences_obs[i] for i in batch_idx]),
                device=self.device, dtype=torch.float32,
            )
            batch_actions = torch.tensor(
                np.array([sequences_actions[i] for i in batch_idx]),
                device=self.device, dtype=torch.float32,
            )
            batch_old_log_probs = torch.tensor(
                np.array([sequences_log_probs[i] for i in batch_idx]),
                device=self.device, dtype=torch.float32,
            )
            batch_advantages = torch.tensor(
                np.array([sequences_advantages[i] for i in batch_idx]),
                device=self.device, dtype=torch.float32,
            )
            batch_returns = torch.tensor(
                np.array([sequences_returns[i] for i in batch_idx]),
                device=self.device, dtype=torch.float32,
            )
            batch_masks = torch.tensor(
                np.array([sequences_masks[i] for i in batch_idx]),
                device=self.device, dtype=torch.float32,
            )

            # Normalize advantages within batch
            valid = batch_masks.bool()
            adv_valid = batch_advantages[valid]
            if len(adv_valid) > 1:
                batch_advantages[valid] = (adv_valid - adv_valid.mean()) / (adv_valid.std() + 1e-8)

            yield {
                "obs": batch_obs,
                "actions": batch_actions,
                "old_log_probs": batch_old_log_probs,
                "advantages": batch_advantages,
                "returns": batch_returns,
                "masks": batch_masks,
            }

    def reset(self):
        self.ptr = 0
        self.full = False

"""Utilities for reward and observation normalization."""

import numpy as np


class RunningMeanStd:
    """Tracks running mean and variance using Welford's algorithm."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # small epsilon to avoid div by zero

    def update(self, batch: np.ndarray):
        batch = np.asarray(batch)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0] if batch.ndim > 0 else 1
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = m2 / total_count
        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)


class RewardNormalizer:
    """
    Normalizes rewards using a running estimate of return statistics.
    Divides rewards by running std of returns (not mean-centered,
    to preserve reward sign).
    """

    def __init__(self, gamma: float = 0.99, clip: float = 10.0):
        self.gamma = gamma
        self.clip = clip
        self.rms = RunningMeanStd(shape=())
        self.ret = 0.0  # running discounted return

    def normalize(self, reward: float, done: bool) -> float:
        self.ret = self.ret * self.gamma + reward
        self.rms.update(np.array([self.ret]))
        if done:
            self.ret = 0.0
        normalized = reward / self.rms.std
        return float(np.clip(normalized, -self.clip, self.clip))