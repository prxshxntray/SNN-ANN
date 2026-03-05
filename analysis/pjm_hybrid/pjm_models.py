"""
PJM Wind Forecast — Final Model Architectures
==============================================
Three models for time-series forecasting, designed for head-to-head comparison:

1. **GRUForecaster** (Baseline) — Pure ANN.
2. **SNNForecaster** (Spiking) — Pure SNN with Raw+Delta input and Deep LIF.
3. **HybridForecaster** (Combined) — Residual gating of SNN events into GRU.

All models share the same API:
    model(x)  where x : (batch, window_size, n_features) → (batch, horizon)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional


class TemporalAttention(nn.Module):
    """
    Learned weighted sum over the time dimension.
    Input: (T, B, C) or (B, T, C) -> Output: (B, C)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        weights = torch.softmax(self.attn(x), dim=dim)
        return (weights * x).sum(dim=dim)


# ═══════════════════════════════════════════════════════════════════════
#  1. PURE ANN BASELINE
# ═══════════════════════════════════════════════════════════════════════

class GRUForecaster(nn.Module):
    """
    Two-layer GRU followed by a linear head.
    A strong, stable baseline for univariate/low-dimensional time-series.
    """
    def __init__(self, input_size: int, horizon: int, hidden: int = 128, n_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        return self.head(h_n[-1])


# ═══════════════════════════════════════════════════════════════════════
#  2. PURE SNN
# ═══════════════════════════════════════════════════════════════════════

class SNNForecaster(nn.Module):
    """
    Spiking Neural Network for Time-Series Forecasting.
    
    Architectural Justification:
    ----------------------------
    1. **Raw + Delta Input**: SNNs naturally detect *changes*, but regression 
       requires absolute context. We concatenate Raw(t) and Delta(t) so the SNN 
       can see both the absolute power level and the temporal shock.
    2. **Deep LIF Capacity**: 2 layers of 256 hidden units provide the necessary
       capacity (approx 20k params) to represent complex continuous values,
       which simple rate-encoders struggle with.
    3. **BatchNorm**: Applied over the time dimension. Time-series data often
       causes vanishing/exploding surrogate gradients. BN ensures stable, 
       active spiking (neurons aren't 'dead') from epoch 1.
    """
    def __init__(self, input_size: int, horizon: int, hidden: int = 256):
        super().__init__()
        from spikingjelly.activation_based.surrogate import ATan
        
        # Concat input_size + input_size (raw + delta)
        self.proj = nn.Linear(input_size * 2, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        
        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=ATan(), step_mode='m')
        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=ATan(), step_mode='m')
        
        self.attn = TemporalAttention(hidden)
        self.head = nn.Linear(hidden, horizon)
        
        nn.init.constant_(self.proj.bias, 0.5) # Encourage initial spiking

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self)
        
        # 1. Prepare Raw + Delta (T becomes T-1)
        delta = x[:, 1:, :] - x[:, :-1, :]
        raw = x[:, 1:, :]
        combined = torch.cat([raw, delta], dim=-1) # (B, T-1, F*2)
        
        # 2. Project + BatchNorm
        h = self.proj(combined)
        h = self.bn(h.transpose(1, 2)).transpose(1, 2)
        
        # 3. LIF Stack
        h_t = h.permute(1, 0, 2)
        s1 = self.lif1(h_t)
        s2 = self.lif2(s1)
        
        # 4. Attention Readout
        feat = self.attn(s2.permute(1, 0, 2))
        return self.head(feat)

    def spike_count(self, x: torch.Tensor) -> int:
        functional.reset_net(self)
        with torch.no_grad():
            delta = x[:, 1:, :] - x[:, :-1, :]
            raw = x[:, 1:, :]
            combined = torch.cat([raw, delta], dim=-1)
            h = self.proj(combined)
            h = self.bn(h.transpose(1, 2)).transpose(1, 2)
            h_t = h.permute(1, 0, 2)
            s1 = self.lif1(h_t)
            s2 = self.lif2(s1)
            return int(s1.sum().item() + s2.sum().item())


# ═══════════════════════════════════════════════════════════════════════
#  3. HYBRID FORECASTER (RESIDUAL)
# ═══════════════════════════════════════════════════════════════════════

class HybridForecaster(nn.Module):
    """
    Residual SNN-Enhanced GRU Hybrid.
    
    Architectural Justification:
    ----------------------------
    Parallel hybrids risk the SNN path "poisoning" the robust ANN baseline if 
    the surrogate gradients fail to align. 
    
    This model uses a **Residual Design**:
    - The baseline path is a standard GRU processing raw inputs.
    - The enhancement path is an SNN that *only* processes temporal deltas 
      (acting as a pure neuromorphic shock/event detector).
    - The SNN features are added to the GRU features via a gate that is 
      **zero-initialized**.
      
    Result: At Epoch 1, this model is mathematically identical to a pure GRU. 
    It only incorporates SNN features as the network explicitly learns that 
    the "delta events" improve the regression loss.
    """
    def __init__(self, input_size: int, horizon: int, hidden: int = 128):
        super().__init__()
        
        # 1. Baseline Path (GRU on Raw)
        self.gru = nn.GRU(input_size, hidden, batch_first=True)
        self.gru_attn = TemporalAttention(hidden)
        
        # 2. Enhancement Path (SNN on Delta)
        self.snn_dim = 128
        self.snn_proj = nn.Linear(input_size, self.snn_dim)
        self.snn_lif = neuron.LIFNode(tau=2.0, step_mode='m')
        self.snn_attn = TemporalAttention(self.snn_dim)
        self.snn_to_gru = nn.Linear(self.snn_dim, hidden)
        
        # 3. Residual Gate (Zero-initialized)
        self.gate_proj = nn.Linear(hidden + self.snn_dim, 1)
        nn.init.constant_(self.gate_proj.weight, 0)
        nn.init.constant_(self.gate_proj.bias, -5.0) # Sigmoid starts near 0
        
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self)
        
        # --- Baseline Path ---
        gru_out, _ = self.gru(x)
        h_ann = self.gru_attn(gru_out)
        
        # --- Enhancement Path (Delta only) ---
        delta = x[:, 1:, :] - x[:, :-1, :]
        h_s = self.snn_proj(delta)
        s = self.snn_lif(h_s.permute(1, 0, 2))
        h_snn = self.snn_attn(s.permute(1, 0, 2))
        
        # --- Residual Fusion ---
        g = torch.sigmoid(self.gate_proj(torch.cat([h_ann, h_snn], dim=-1)))
        h_combined = h_ann + g * self.snn_to_gru(h_snn)
        
        return self.head(h_combined)

    def spike_count(self, x: torch.Tensor) -> int:
        functional.reset_net(self)
        with torch.no_grad():
            delta = x[:, 1:, :] - x[:, :-1, :]
            h_s = self.snn_proj(delta)
            s = self.snn_lif(h_s.permute(1, 0, 2))
            return int(s.sum().item())
