"""
PJM Wind Forecast — Model Architectures (V5 + V6)
===================================================
Time-series forecasting models for head-to-head comparison:

V5 Models (Baseline):
  1. **GRUForecaster** — Pure ANN baseline.
  2. **SNNForecaster** — Pure SNN with Raw+Delta input and Deep LIF.
  3. **HybridForecaster** — Residual gating of SNN events into GRU.

V6 Models (Literature-Informed, see architecture_evolution.md Phase 4):
  4. **SNNForecasterV6** — Conv Spike Encoder, Temporal Alignment, BN, SpikingAttention.
  5. **HybridForecasterV6** — Conv Spike Encoder → Downsample → GRU with SpikingAttention.

Supporting Modules:
  - **ConvSpikeEncoder** — Learned convolutional encoding (SeqSNN, ICML 2024)
  - **SpikingAttention** — Sigmoid-gated Q·K^T attention (SpikSTAG / Spikeformer)

All models share the same API:
    model(x)  where x : (batch, window_size, n_features) → (batch, horizon)
    V6 models additionally support: model(x, return_trajectory=True) for DecodingLoss
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
        # V5: PLIF Neurons (learn_beta=True) allows neurons to learn their own leak rates
        self.lif1 = snn.Leaky(beta=torch.full((hidden,), 0.9), learn_beta=True, spike_grad=spike_grad, init_hidden=True) #creates the first PLIF neuron layer
        self.lif2 = snn.Leaky(beta=torch.full((hidden,), 0.9), learn_beta=True, spike_grad=spike_grad, init_hidden=True) #second PLIF neuron stacked on top of first
        
        # V5 Continuous Readout Layer (also PLIF)
        self.lif3 = snn.Leaky(beta=torch.full((hidden,), 0.9), learn_beta=True, spike_grad=spike_grad, init_hidden=True, output=True)
        
        self.attn = TemporalAttention(hidden) #learns which timesteps are most important in the spike sequence
        self.head = nn.Linear(hidden, horizon) #final linear layer converting the learned represenation into forecast ouputs
        
        nn.init.constant_(self.proj.bias, 0.5) # Encourage initial spiking during early training

    def update_surrogate_slope(self, new_slope: float):
        from snntorch import surrogate
        spike_grad = surrogate.fast_sigmoid(slope=new_slope)
        self.lif1.spike_grad = spike_grad
        self.lif2.spike_grad = spike_grad
        self.lif3.spike_grad = spike_grad


    def forward(self, x: torch.Tensor) -> torch.Tensor: #x represents a time-series batch
        import snntorch as snn
        from snntorch import utils
        utils.reset(self) #resets membrane potentials of all spiking neurons or else the next batch starts with leftover voltage
        # from the previous batch which causes info leakage between samples.
        
        # Clamp learnable beta to mathematically safe ranges (prevent exponential explosion)
        self.lif1.beta.data.clamp_(0.0, 0.99)
        self.lif2.beta.data.clamp_(0.0, 0.99)
        self.lif3.beta.data.clamp_(0.0, 0.99)
        
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
    Feature-Extraction SNN-GRU Hybrid.
    
    Architectural Justification:
    ----------------------------
    V5 shifts from a parallel Residual design to a chained Feature-Extraction paradigm.
    - SNNs excel at early spatio-temporal dynamics (event detection).
    - GRUs excel at deep continuous regression.
    
    This model uses the SNN to process the raw input deltas and output a sequence 
    of rich "membrane potential features". The GRU takes those SNN features 
    (concatenated with the raw features) as its formal input, acting as the 
    ultimate continuous regression head.
    """
    def __init__(self, input_size: int, horizon: int, hidden: int = 128):
        super().__init__()
        import snntorch as snn
        from snntorch import surrogate
        
        # 1. Enhancement Path (SNN on Delta)
        self.snn_dim = 128 # hidden dimension for SNN and this defines spike feature dimension
        self.snn_proj = nn.Linear(input_size, self.snn_dim) #projects delta features into the spike feature space
        
        spike_grad = surrogate.fast_sigmoid(slope=25)
        # V5: PLIF Feature extractors
        self.snn_lif1 = snn.Leaky(beta=torch.full((self.snn_dim,), 0.9), learn_beta=True, spike_grad=spike_grad, init_hidden=True) #LIF neuron integrates signal and emits spikes
        self.snn_lif2 = snn.Leaky(beta=torch.full((self.snn_dim,), 0.9), learn_beta=True, spike_grad=spike_grad, init_hidden=True, output=True) # Reads out membrane potential
        
        # 2. Chained GRU Path
        self.gru = nn.GRU(input_size + self.snn_dim, hidden, batch_first=True)
        self.gru_attn = TemporalAttention(hidden) #applies attention across time to extract a single sequence representation
        
        self.head = nn.Linear(hidden, horizon)

    def update_surrogate_slope(self, new_slope: float):
        from snntorch import surrogate
        spike_grad = surrogate.fast_sigmoid(slope=new_slope)
        self.snn_lif1.spike_grad = spike_grad
        self.snn_lif2.spike_grad = spike_grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import snntorch as snn
        from snntorch import utils
        utils.reset(self)
        
        # Clamp learnable beta to mathematically safe ranges
        self.snn_lif1.beta.data.clamp_(0.0, 0.99)
        self.snn_lif2.beta.data.clamp_(0.0, 0.99)
        
        # --- Enhancement Path (Delta only into SNN) ---
        delta = x[:, 1:, :] - x[:, :-1, :]
        raw = x[:, 1:, :]
        
        h_s = self.snn_proj(delta)
        h_s_t = h_s.permute(1, 0, 2)
        
        mem_rec = []
        for step in range(h_s_t.size(0)):
            spk1 = self.snn_lif1(h_s_t[step])
            _, mem2 = self.snn_lif2(spk1)
            mem_rec.append(mem2)
        mem_rec = torch.stack(mem_rec, dim=0)
        
        h_snn = mem_rec.permute(1, 0, 2) # (B, T, snn_dim)
        
        # --- Chained Baseline Path ---
        h_combined = torch.cat([raw, h_snn], dim=-1) # (B, T, input_size + snn_dim)
        
        gru_out, _ = self.gru(h_combined)
        h_ann = self.gru_attn(gru_out)
        
        return self.head(h_ann)

    def spike_count(self, x: torch.Tensor) -> int:
        import snntorch as snn
        from snntorch import utils
        utils.reset(self)
        with torch.no_grad():
            delta = x[:, 1:, :] - x[:, :-1, :]
            h_s = self.snn_proj(delta)
            h_s_t = h_s.permute(1, 0, 2)
            
            total_spikes = 0
            self.snn_lif1.output = False
            self.snn_lif2.output = False
            for step in range(h_s_t.size(0)):
                spk1 = self.snn_lif1(h_s_t[step])
                spk2 = self.snn_lif2(spk1)
                total_spikes += int(spk1.sum().item() + spk2.sum().item())
            self.snn_lif1.output = False # default
            self.snn_lif2.output = True
            
            return total_spikes


# ═══════════════════════════════════════════════════════════════════════
#  V6 ARCHITECTURES — Literature-Informed Improvements
# ═══════════════════════════════════════════════════════════════════════
# Based on:
#   Paper 1: SpikSTAG (Dual-Path Fusion, SSA)
#   Paper 2: Spikeformer (Conv Tokenizer, STA)
#   Paper 3: Lucas & Portillo (PWM encoding, spike-timing loss)
#   Paper 4: Manna et al. (Derivative encoding, DecodingLoss)
#   Paper 6: SeqSNN / ICML 2024 (Conv Spike Encoder, Temporal Alignment,
#            Spike-RNN, BN before spiking, Beta sensitivity)
#   Paper 8: Khoee et al. (Sparsity control, lateral inhibition)
# ═══════════════════════════════════════════════════════════════════════


class ConvSpikeEncoder(nn.Module):
    """
    Convolutional Spike Encoder from SeqSNN (Lv et al., ICML 2024).
    
    Replaces manual raw+delta with a *learned* encoding that captures
    morphological (shape) features in subsequences. Paper showed +0.09 R²
    improvement over delta encoding.
    
    Architecture:
        Conv1d(input → hidden, kernel_size) → BatchNorm → LIF
    
    The conv kernel learns to detect temporal shapes (ramps, dips, spikes)
    rather than just point-to-point differences.
    
    Temporal Alignment:
        Each data timestep generates Tₛ spiking events via the conv encoder,
        giving the SNN finer temporal granularity (ΔT = Tₛ · Δt).
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        spiking_steps: int = 4,
        kernel_size: int = 3,
    ):
        super().__init__()
        import snntorch as snn
        from snntorch import surrogate
        
        self.spiking_steps = spiking_steps
        self.hidden_size = hidden_size
        
        # Conv encoder expands each timestep into spiking_steps channels
        # Paper 6: S = SN(BN(Conv(X)))
        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size * spiking_steps,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # same-padding
            groups=1,
        )
        self.bn = nn.BatchNorm1d(hidden_size * spiking_steps)
        
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.lif = snn.Leaky(
            beta=torch.full((hidden_size,), 0.9),
            learn_beta=True,
            spike_grad=spike_grad,
            init_hidden=True,
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, input_size) — raw time-series input
        Returns:
            spikes: (Tₛ*T, B, hidden) — spike train
            mem:    (Tₛ*T, B, hidden) — membrane potentials
        """
        import snntorch as snn
        from snntorch import utils
        
        B, T, C = x.shape
        
        # Conv1d expects (B, C, T)
        h = self.conv(x.transpose(1, 2))      # (B, hidden*Ts, T)
        h = self.bn(h)                          # BatchNorm before spiking
        h = h.transpose(1, 2)                   # (B, T, hidden*Ts)
        
        # Reshape to (B, T, Ts, hidden) → (B, T*Ts, hidden) → (T*Ts, B, hidden)
        h = h.view(B, T, self.spiking_steps, self.hidden_size)
        h = h.reshape(B, T * self.spiking_steps, self.hidden_size)
        h = h.permute(1, 0, 2)  # (T*Ts, B, hidden)
        
        spk_rec, mem_rec = [], []
        for step in range(h.size(0)):
            spk = self.lif(h[step])
            spk_rec.append(spk)
            mem_rec.append(self.lif.mem)        # access internal membrane state
        
        return torch.stack(spk_rec), torch.stack(mem_rec)


class SpikingAttention(nn.Module):
    """
    Spiking Self-Attention (SSA) — lightweight version.
    
    Inspired by SpikSTAG (Paper 1) and Spikeformer (Paper 2).
    Uses spike-based Q·K^T without softmax for hardware friendliness.
    For V6 we use a simplified version: learned temporal weighting
    with batch normalization for stability.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.scale = dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)
        Returns:
            out: (B, C) — attended summary
        """
        Q = self.q_proj(x)  # (B, T, C)
        K = self.k_proj(x)  # (B, T, C)
        V = self.v_proj(x)  # (B, T, C)
        
        # Scaled dot-product attention (spike-friendly: no softmax exp)
        # Use sigmoid instead of softmax for bounded, spike-compatible attention
        attn = torch.sigmoid(torch.bmm(Q, K.transpose(1, 2)) * self.scale)  # (B, T, T)
        attended = torch.bmm(attn, V)  # (B, T, C)
        
        # Pool over time
        out = attended.mean(dim=1)  # (B, C)
        out = self.bn(out)
        return out


# ═══════════════════════════════════════════════════════════════════════
#  4. V6 PURE SNN
# ═══════════════════════════════════════════════════════════════════════

class SNNForecasterV6(nn.Module):
    """
    V6 Spiking Neural Network for Time-Series Forecasting.
    
    Improvements over V5:
    ---------------------
    1. **Conv Spike Encoder** (Paper 6): Replaces raw+delta with learned
       convolutional encoding that captures temporal morphology.
    2. **Temporal Alignment** (Paper 6): Tₛ spiking sub-steps per data
       timestep for finer temporal resolution.
    3. **Batch Norm before ALL LIF layers** (Paper 6): Prevents dead neurons
       and stabilizes surrogate gradients throughout the network.
    4. **Spiking Attention** (Papers 1,2): Upgraded attention mechanism
       with Q/K/V projections and sigmoid-gated weighting.
    5. **Membrane trajectory output** (Paper 4): Exposes full membrane
       sequences for DecodingLoss training.
    """
    def __init__(
        self,
        input_size: int,
        horizon: int,
        hidden: int = 256,
        spiking_steps: int = 4,
    ):
        super().__init__()
        import snntorch as snn
        from snntorch import surrogate
        
        self.hidden = hidden
        self.spiking_steps = spiking_steps
        
        # V6: Convolutional Spike Encoder (replaces manual raw+delta)
        self.encoder = ConvSpikeEncoder(
            input_size=input_size,
            hidden_size=hidden,
            spiking_steps=spiking_steps,
        )
        
        # V6: BN before each LIF layer
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # Deep LIF stack with learnable leak (PLIF)
        self.lif2 = snn.Leaky(
            beta=torch.full((hidden,), 0.9), learn_beta=True,
            spike_grad=spike_grad, init_hidden=True,
        )
        self.lif3 = snn.Leaky(
            beta=torch.full((hidden,), 0.9), learn_beta=True,
            spike_grad=spike_grad, init_hidden=True, output=True,
        )
        
        # V6: Spiking Attention (upgraded from V5 TemporalAttention)
        self.attn = SpikingAttention(hidden)
        self.head = nn.Linear(hidden, horizon)
    
    def update_surrogate_slope(self, new_slope: float):
        from snntorch import surrogate
        spike_grad = surrogate.fast_sigmoid(slope=new_slope)
        self.encoder.lif.spike_grad = spike_grad
        self.lif2.spike_grad = spike_grad
        self.lif3.spike_grad = spike_grad
    
    def forward(self, x: torch.Tensor, return_trajectory: bool = False):
        """
        Args:
            x: (B, T, input_size)
            return_trajectory: If True, return membrane trajectory for DecodingLoss
        Returns:
            predictions: (B, horizon)
            membrane_trajectory: (T_total, B, hidden) — only if return_trajectory=True
        """
        import snntorch as snn
        from snntorch import utils
        utils.reset(self)
        
        # Clamp learnable betas
        self.encoder.lif.beta.data.clamp_(0.0, 0.99)
        self.lif2.beta.data.clamp_(0.0, 0.99)
        self.lif3.beta.data.clamp_(0.0, 0.99)
        
        # 1. Conv Spike Encoder → (T*Ts, B, hidden)
        spk_enc, mem_enc = self.encoder(x)
        
        # 2. Deep LIF stack with BN
        mem_rec = []
        spk_count = 0
        T_total = spk_enc.size(0)
        
        for step in range(T_total):
            # BN before spiking (Paper 6)
            h = self.bn2(spk_enc[step])
            s2 = self.lif2(h)
            spk_count += s2.sum().item()
            
            h = self.bn3(s2)
            _, mem3 = self.lif3(h)
            mem_rec.append(mem3)
        
        mem_rec = torch.stack(mem_rec, dim=0)  # (T_total, B, hidden)
        
        # 3. Spiking Attention Readout
        feat = self.attn(mem_rec.permute(1, 0, 2))  # (B, hidden)
        predictions = self.head(feat)
        
        if return_trajectory:
            return predictions, mem_rec
        return predictions
    
    def spike_count(self, x: torch.Tensor) -> int:
        import snntorch as snn
        from snntorch import utils
        utils.reset(self)
        with torch.no_grad():
            spk_enc, _ = self.encoder(x)
            total_spikes = int(spk_enc.sum().item())
            
            T_total = spk_enc.size(0)
            for step in range(T_total):
                h = self.bn2(spk_enc[step])
                s2 = self.lif2(h)
                total_spikes += int(s2.sum().item())
                h = self.bn3(s2)
                s3 = self.lif3(h)
                total_spikes += int(s3.sum().item()) if isinstance(s3, torch.Tensor) else 0
            return total_spikes


# ═══════════════════════════════════════════════════════════════════════
#  5. V6 HYBRID FORECASTER
# ═══════════════════════════════════════════════════════════════════════

class HybridForecasterV6(nn.Module):
    """
    V6 Feature-Extraction SNN-GRU Hybrid.
    
    Improvements over V5:
    ---------------------
    1. **Conv Spike Encoder** for the SNN path (replaces manual delta)
    2. **BN throughout** for stable training
    3. **Temporal Alignment** with Tₛ sub-steps for richer spike features
    4. **Downsampling** to match GRU's expected temporal resolution
    5. **Membrane trajectory** exposed for DecodingLoss
    
    Architecture:
        Input → Conv Spike Encoder → [spk features] → Downsample → concat(raw) → GRU → Head
    """
    def __init__(
        self,
        input_size: int,
        horizon: int,
        hidden: int = 128,
        snn_dim: int = 128,
        spiking_steps: int = 4,
    ):
        super().__init__()
        import snntorch as snn
        from snntorch import surrogate
        
        self.snn_dim = snn_dim
        self.spiking_steps = spiking_steps
        
        # 1. SNN Enhancement Path with Conv Encoder
        self.encoder = ConvSpikeEncoder(
            input_size=input_size,
            hidden_size=snn_dim,
            spiking_steps=spiking_steps,
        )
        
        # Second LIF layer for deeper spike processing
        self.bn_snn2 = nn.BatchNorm1d(snn_dim)
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.snn_lif2 = snn.Leaky(
            beta=torch.full((snn_dim,), 0.9), learn_beta=True,
            spike_grad=spike_grad, init_hidden=True, output=True,
        )
        
        # Downsample: average Tₛ sub-steps back to original T resolution
        # so SNN features can be concatenated with raw input for GRU
        
        # 2. Chained GRU Path
        self.gru = nn.GRU(input_size + snn_dim, hidden, batch_first=True)
        self.gru_attn = SpikingAttention(hidden)
        
        self.head = nn.Linear(hidden, horizon)
    
    def update_surrogate_slope(self, new_slope: float):
        from snntorch import surrogate
        spike_grad = surrogate.fast_sigmoid(slope=new_slope)
        self.encoder.lif.spike_grad = spike_grad
        self.snn_lif2.spike_grad = spike_grad
    
    def forward(self, x: torch.Tensor, return_trajectory: bool = False):
        """
        Args:
            x: (B, T, input_size)
            return_trajectory: If True, return membrane trajectory
        Returns:
            predictions: (B, horizon)
            membrane_trajectory: optional (T*Ts, B, snn_dim)
        """
        import snntorch as snn
        from snntorch import utils
        utils.reset(self)
        
        B, T, C = x.shape
        
        # Clamp betas
        self.encoder.lif.beta.data.clamp_(0.0, 0.99)
        self.snn_lif2.beta.data.clamp_(0.0, 0.99)
        
        # --- SNN Enhancement Path ---
        spk_enc, mem_enc = self.encoder(x)  # (T*Ts, B, snn_dim)
        
        # Second LIF layer
        mem_rec = []
        T_total = spk_enc.size(0)
        for step in range(T_total):
            h = self.bn_snn2(spk_enc[step])
            _, mem2 = self.snn_lif2(h)
            mem_rec.append(mem2)
        
        mem_rec = torch.stack(mem_rec, dim=0)  # (T*Ts, B, snn_dim)
        
        # Downsample: average Tₛ sub-steps → (T, B, snn_dim)
        mem_down = mem_rec.permute(1, 0, 2)  # (B, T*Ts, snn_dim)
        mem_down = mem_down.reshape(B, T, self.spiking_steps, self.snn_dim)
        mem_down = mem_down.mean(dim=2)  # (B, T, snn_dim)
        
        # --- Chained GRU Path ---
        h_combined = torch.cat([x, mem_down], dim=-1)  # (B, T, input_size + snn_dim)
        
        gru_out, _ = self.gru(h_combined)
        h_ann = self.gru_attn(gru_out)
        
        predictions = self.head(h_ann)
        
        if return_trajectory:
            return predictions, mem_rec
        return predictions
    
    def spike_count(self, x: torch.Tensor) -> int:
        import snntorch as snn
        from snntorch import utils
        utils.reset(self)
        with torch.no_grad():
            spk_enc, _ = self.encoder(x)
            total_spikes = int(spk_enc.sum().item())
            
            T_total = spk_enc.size(0)
            for step in range(T_total):
                h = self.bn_snn2(spk_enc[step])
                spk2, _ = self.snn_lif2(h)
                total_spikes += int(spk2.sum().item()) if isinstance(spk2, torch.Tensor) else 0
            return total_spikes
