"""
PJM Wind Forecast — Custom Loss Functions (V6)
===============================================
Literature-informed loss functions for SNN time-series forecasting.

Based on:
  - Manna et al. (2024) — DecodingLoss: signal-reconstruction loss from spikes
  - Lucas & Portillo (2024) — Spike-timing difference loss (PWM-inspired)
  - Lv et al. (ICML 2024) — SeqSNN training practices

Key insight: Standard MSE on membrane potentials forces the SNN to learn
an exact continuous mapping. DecodingLoss instead optimizes in the
*reconstructed signal domain*, giving the SNN more freedom in how it
represents values (many spike patterns can decode to the same value).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DecodingLoss(nn.Module):
    """
    Signal-reconstruction loss from Manna et al. (2024).

    Instead of comparing raw membrane potentials or spike trains,
    this loss:
      1. Reconstructs the output signal from spike outputs via
         cumulative sum weighted by learned thresholds
      2. Computes MSE in the original signal domain

    This allows the SNN to learn spike patterns that decode to the
    correct value, even if the exact spike timings differ from target.

    For regression (our case), we adapt this by comparing the
    cumulative membrane dynamics rather than binary spikes:
      - Compute a running integral of the membrane potential sequence
      - Compare the trajectory shape, not just final values
    """
    def __init__(self, trajectory_weight: float = 0.5):
        super().__init__()
        self.trajectory_weight = trajectory_weight
        self.mse = nn.MSELoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        membrane_sequence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B, horizon) — final forecast output
            targets:     (B, horizon) — ground truth
            membrane_sequence: (T, B, hidden) — optional membrane potential
                              trajectory from the SNN layers. If provided,
                              adds a trajectory-consistency term.
        Returns:
            Combined loss scalar.
        """
        # Primary: forecast accuracy in original domain
        forecast_loss = self.mse(predictions, targets)

        if membrane_sequence is None or self.trajectory_weight == 0.0:
            return forecast_loss

        # Trajectory term: encourage smooth, monotonically informative
        # membrane evolution by penalizing erratic jumps
        # Compute temporal differences of membrane potential
        mem_diff = membrane_sequence[1:] - membrane_sequence[:-1]  # (T-1, B, H)

        # Penalize the variance of temporal changes (prefer smooth trajectories)
        trajectory_var = mem_diff.var(dim=0).mean()

        return forecast_loss + self.trajectory_weight * trajectory_var


class SpikeRegularizationLoss(nn.Module):
    """
    Spike activity regularization inspired by SeqSNN (Lv et al., ICML 2024).

    Encourages a target firing rate to prevent dead neurons (rate too low)
    or saturated neurons (rate too high). This is critical for maintaining
    meaningful spike-based representations.

    Also inspired by the sparsity control from Khoee et al. (2023) where
    ~15% neuron activation was optimal for memory encoding.
    """
    def __init__(self, target_rate: float = 0.15, strength: float = 0.01):
        super().__init__()
        self.target_rate = target_rate
        self.strength = strength

    def forward(self, spike_counts: torch.Tensor, total_possible: int) -> torch.Tensor:
        """
        Args:
            spike_counts: Total spikes in a forward pass
            total_possible: Total possible spike events (T * B * hidden)
        Returns:
            Regularization loss scalar.
        """
        actual_rate = spike_counts / max(total_possible, 1)
        return self.strength * (actual_rate - self.target_rate) ** 2


class CombinedForecastLoss(nn.Module):
    """
    Unified loss combining forecast MSE, trajectory consistency,
    and spike regularization.

    Usage:
        loss_fn = CombinedForecastLoss(alpha=0.7, trajectory_weight=0.2, spike_reg=0.01)
        loss = loss_fn(preds, targets, membrane_seq, spike_count, total_possible)
    """
    def __init__(
        self,
        alpha: float = 0.8,
        trajectory_weight: float = 0.3,
        spike_reg_strength: float = 0.005,
        target_spike_rate: float = 0.15,
    ):
        super().__init__()
        self.alpha = alpha  # weight for primary MSE
        self.decoding_loss = DecodingLoss(trajectory_weight=trajectory_weight)
        self.spike_reg = SpikeRegularizationLoss(
            target_rate=target_spike_rate,
            strength=spike_reg_strength,
        )
        self.mse = nn.MSELoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        membrane_sequence: torch.Tensor | None = None,
        spike_count: float = 0.0,
        total_possible: int = 1,
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            loss: Combined scalar loss
            components: Dict of individual loss components for logging
        """
        # 1. Primary MSE
        mse_loss = self.mse(predictions, targets)

        # 2. Decoding / trajectory loss
        decoding_loss = self.decoding_loss(predictions, targets, membrane_sequence)

        # 3. Spike regularization
        reg_loss = self.spike_reg(
            torch.tensor(spike_count, dtype=torch.float32, device=predictions.device),
            total_possible,
        )

        # Combine
        total = self.alpha * mse_loss + (1 - self.alpha) * decoding_loss + reg_loss

        components = {
            "mse": mse_loss.item(),
            "decoding": decoding_loss.item(),
            "spike_reg": reg_loss.item(),
            "total": total.item(),
        }

        return total, components
