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
    4. **V4 Membrane Readout**: For continuous regression, the final layer reads
       out membrane voltage instead of discrete binary spikes.
    """
    def __init__(self, input_size: int, horizon: int, hidden: int = 256):
        super().__init__()
        import snntorch as snn
        from snntorch import surrogate #ATan surrogate gradient
        
        # Concat input_size + input_size (raw + delta)
        self.proj = nn.Linear(input_size * 2, hidden) # input_size*2 because model later concatenates raw values and delta values
        self.bn = nn.BatchNorm1d(hidden)
        
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True) #creates the first LIF neuron layer
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True) #second LIF neuron stacked on top of first
        
        # V4 Continuous Readout Layer
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True)
        
        self.attn = TemporalAttention(hidden) #learns which timesteps are most important in the spike sequence
        self.head = nn.Linear(hidden, horizon) #final linear layer converting the learned represenation into forecast ouputs
        
        nn.init.constant_(self.proj.bias, 0.5) # Encourage initial spiking during early training

    def forward(self, x: torch.Tensor) -> torch.Tensor: #x represents a time-series batch
        import snntorch as snn
        from snntorch import utils
        utils.reset(self) #resets membrane potentials of all spiking neurons or else the next batch starts with leftover voltage
        # from the previous batch which causes info leakage between samples.
        
        # 1. Prepare Raw + Delta (T becomes T-1)
        delta = x[:, 1:, :] - x[:, :-1, :] #computes changes between consecutive timesteps. x : (B, T, F)
        raw = x[:, 1:, :] #removes first timestep. raw and delta have the same length because the first timestep has no value and it cannot produce a delta.
        combined = torch.cat([raw, delta], dim=-1) # (B, T-1, F*2) --> feat dimension doubles and now each timestep contains [value , change]
        
        # 2. Project + BatchNorm
        h = self.proj(combined) #projects features into hidden dimension wich is standard for neural feature dimension
        h = self.bn(h.transpose(1, 2)).transpose(1, 2)
        
        # 3. LIF Stack
        h_t = h.permute(1, 0, 2)
        
        mem_rec = []
        for step in range(h_t.size(0)):
            s1 = self.lif1(h_t[step]) #integrates input, fires spike if threshold crossed, leaks voltage over time
            s2 = self.lif2(s1) #processes spikes again to learn higher-level temporal spike patterns
            _, mem3 = self.lif3(s2) # V4: Read membrane continuous voltage for regression
            mem_rec.append(mem3)
            
        mem_rec = torch.stack(mem_rec, dim=0) # (T, B, hidden)
        
        # 4. Attention Readout
        feat = self.attn(mem_rec.permute(1, 0, 2)) #attention learns which timesteps matter most AFTER converting back to batch format
        return self.head(feat) #Linear layer maps representation to (B, horizon) --> E.g. predict next 6 timesteps for that batch size

    def spike_count(self, x: torch.Tensor) -> int:
        import snntorch as snn
        from snntorch import utils
        utils.reset(self)
        with torch.no_grad():
            delta = x[:, 1:, :] - x[:, :-1, :]
            raw = x[:, 1:, :]
            combined = torch.cat([raw, delta], dim=-1)
            h = self.proj(combined)
            h = self.bn(h.transpose(1, 2)).transpose(1, 2)
            h_t = h.permute(1, 0, 2)
            
            total_spikes = 0
            for step in range(h_t.size(0)):
                s1 = self.lif1(h_t[step])
                s2 = self.lif2(s1)
                total_spikes += int(s1.sum().item() + s2.sum().item())
            return total_spikes #aft running through LIF layers, count spikes emitted by each layer


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
        import snntorch as snn
        from snntorch import surrogate
        
        # 1. Baseline Path (GRU on Raw)
        self.gru = nn.GRU(input_size, hidden, batch_first=True)  # Baseline Path (GRU on Raw)
        self.gru_attn = TemporalAttention(hidden) #applies attention across time to extract a single sequence representation
        
        # 2. Enhancement Path (SNN on Delta)
        self.snn_dim = 128 # hidden dimension for SNN and this defines spike feature dimension
        self.snn_proj = nn.Linear(input_size, self.snn_dim) #projects delta features into the spike feature space
        
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.snn_lif = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True) #LIF neuron integrates signal and emits spikes when threshold crossed
        
        self.snn_attn = TemporalAttention(self.snn_dim) #Learns which spike moments matter most
        self.snn_to_gru = nn.Linear(self.snn_dim, hidden) #needed for both paths to share the same dimension
        
        # 3. Residual Gate (Zero-initialized)
        self.gate_proj = nn.Linear(hidden + self.snn_dim, 1) #creates a gating signal controlling how much SNN information to use
        nn.init.constant_(self.gate_proj.weight, 0)
        nn.init.constant_(self.gate_proj.bias, -5.0) # Sigmoid starts near 0 at the start of training and SNN contribution gradually grows during training
        
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import snntorch as snn
        from snntorch import utils
        utils.reset(self)
        
        # --- Baseline Path ---
        gru_out, _ = self.gru(x)
        h_ann = self.gru_attn(gru_out)
        
        # --- Enhancement Path (Delta only) ---
        delta = x[:, 1:, :] - x[:, :-1, :]
        h_s = self.snn_proj(delta)
        h_s_t = h_s.permute(1, 0, 2)
        
        mem_rec = []
        for step in range(h_s_t.size(0)):
            _, mem = self.snn_lif(h_s_t[step])
            mem_rec.append(mem)
        mem_rec = torch.stack(mem_rec, dim=0)
        
        h_snn = self.snn_attn(mem_rec.permute(1, 0, 2))
        
        # --- Residual Fusion ---
        g = torch.sigmoid(self.gate_proj(torch.cat([h_ann, h_snn], dim=-1))) #signmoid determines if we use SNN or ignore fully. 0 is ignore and 1 is fully use.
        h_combined = h_ann + g * self.snn_to_gru(h_snn) #residual fusion
        
        return self.head(h_combined)

    def spike_count(self, x: torch.Tensor) -> int:
        import snntorch as snn
        from snntorch import utils
        utils.reset(self)
        with torch.no_grad():
            delta = x[:, 1:, :] - x[:, :-1, :]
            h_s = self.snn_proj(delta)
            h_s_t = h_s.permute(1, 0, 2)
            
            total_spikes = 0
            self.snn_lif.output = False
            for step in range(h_s_t.size(0)):
                spk = self.snn_lif(h_s_t[step])
                total_spikes += int(spk.sum().item())
            self.snn_lif.output = True
            
            return total_spikes
