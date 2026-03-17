import torch
import torch.nn as nn
import snntorch as snn
from snntorch import utils
import numpy as np
import os
import time

# ==============================================================================
#  UTILITIES & COMPONENTS
# ==============================================================================

class TemporalAttention(nn.Module):
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

# ==============================================================================
#  TIER 1 MODELS (V1 - V5)
# ==============================================================================

class GRUForecaster(nn.Module):
    def __init__(self, input_size, horizon, hidden=128, n_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, n_layers, batch_first=True)
        self.head = nn.Linear(hidden, horizon)
    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])

class SNNForecaster(nn.Module):
    def __init__(self, input_size, horizon, hidden=256):
        super().__init__()
        self.hidden = hidden
        self.lif1 = snn.Leaky(beta=0.9, learn_beta=True)
        self.fc1 = nn.Linear(input_size, hidden)
        self.head = nn.Linear(hidden, horizon)
        
    def forward(self, x):
        B, T, C = x.shape
        mem = torch.zeros(B, self.hidden, device=x.device)
        for t in range(T):
            spk, mem = self.lif1(self.fc1(x[:, t, :]), mem)
        return self.head(mem)

class HybridForecaster(nn.Module):
    def __init__(self, input_size, horizon, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.snn_fc = nn.Linear(input_size, hidden)
        self.snn_lif = snn.Leaky(beta=0.9)
        self.gru = nn.GRU(input_size + hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        B, T, C = x.shape
        mem = torch.zeros(B, self.hidden, device=x.device)
        snn_outs = []
        for t in range(T):
            spk, mem = self.snn_lif(self.snn_fc(x[:, t, :]), mem)
            snn_outs.append(spk)
        h_comb = torch.cat([x, torch.stack(snn_outs, dim=1)], dim=-1)
        _, h_gru = self.gru(h_comb)
        return self.head(h_gru[-1])

# ==============================================================================
#  TIER 2 COMPONENTS (V6 - V7)
# ==============================================================================

class ConvSpikeEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, spiking_steps=4):
        super().__init__()
        self.spiking_steps = spiking_steps
        self.hidden_size = hidden_size
        self.conv = nn.Conv1d(input_size, hidden_size * spiking_steps, kernel_size=1)
        self.bn = nn.BatchNorm1d(hidden_size * spiking_steps)
        self.lif = snn.Leaky(beta=0.9, learn_beta=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        B, T, C = x.shape
        h = self.bn(self.conv(x.transpose(1, 2))).transpose(1, 2)
        h = h.reshape(B, T * self.spiking_steps, self.hidden_size).permute(1, 0, 2)
        
        mem = torch.zeros(B, self.hidden_size, device=x.device)
        spk_rec, mem_rec, sum_spks = [], [], 0.0
        for t in range(h.size(0)):
            spk, mem = self.lif(h[t], mem)
            spk_rec.append(spk); mem_rec.append(mem)
            sum_spks += spk.detach().sum().item()
        return torch.stack(spk_rec), torch.stack(mem_rec), sum_spks

class SpikingAttentionSSA(nn.Module):
    """Refined Spiking Attention (SSA)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scale = dim**-0.5
        self.q = nn.Linear(dim, dim); self.bnq = nn.BatchNorm1d(dim); self.lq = snn.Leaky(beta=0.9)
        self.k = nn.Linear(dim, dim); self.bnk = nn.BatchNorm1d(dim); self.lk = snn.Leaky(beta=0.9)
        self.v = nn.Linear(dim, dim); self.bnv = nn.BatchNorm1d(dim); self.lv = snn.Leaky(beta=0.9)

    def forward(self, x):
        B, T, C = x.shape
        xf = x.reshape(B*T, C)
        qt = self.bnq(self.q(xf)).reshape(B, T, C).permute(1,0,2)
        kt = self.bnk(self.k(xf)).reshape(B, T, C).permute(1,0,2)
        vt = self.bnv(self.v(xf)).reshape(B, T, C).permute(1,0,2)
        
        mq, mk, mv = [torch.zeros(B, self.dim, device=x.device) for _ in range(3)]
        qr, kr, vr, sc = [], [], [], 0.0
        for t in range(T):
            s, mq = self.lq(qt[t], mq); qr.append(s); sc += s.detach().sum().item()
            s, mk = self.lk(kt[t], mk); kr.append(s); sc += s.detach().sum().item()
            s, mv = self.lv(vt[t], mv); vr.append(s); sc += s.detach().sum().item()
            
        Q, K, V = torch.stack(qr).permute(1,0,2), torch.stack(kr).permute(1,0,2), torch.stack(vr).permute(1,0,2)
        attn = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) * self.scale, dim=-1)
        return torch.bmm(attn, V).mean(dim=1), sc

class SpikingAttention(SpikingAttentionSSA):
    """Alias for V6 compatibility"""
    def forward(self, x):
        out, _ = super().forward(x)
        return out

class SNNForecasterV6(nn.Module):
    def __init__(self, input_size, horizon, hidden=256):
        super().__init__()
        self.hidden = hidden
        self.encoder = ConvSpikeEncoder(input_size, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.lif2 = snn.Leaky(beta=0.9, learn_beta=True)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.lif3 = snn.Leaky(beta=0.9, learn_beta=True)
        self.attn = SpikingAttention(hidden)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x, return_trajectory=False):
        spk_enc, _, sum_spks = self.encoder(x)
        B = x.size(0)
        m2 = torch.zeros(B, self.hidden, device=x.device)
        m3 = torch.zeros(B, self.hidden, device=x.device)
        mem_rec = []
        for t in range(spk_enc.size(0)):
            s2, m2 = self.lif2(self.bn2(spk_enc[t]), m2)
            sum_spks += s2.detach().sum().item()
            s3, m3 = self.lif3(self.bn3(s2), m3)
            mem_rec.append(m3)
            sum_spks += s3.detach().sum().item()
        
        feat = self.attn(torch.stack(mem_rec).permute(1, 0, 2))
        preds = self.head(feat)
        if return_trajectory: return preds, torch.stack(mem_rec), sum_spks
        return preds, sum_spks

class HybridForecasterV6(nn.Module):
    def __init__(self, input_size, horizon, snn_dim=128, ann_dim=128):
        super().__init__()
        self.snn_dim = snn_dim
        self.encoder = ConvSpikeEncoder(input_size, snn_dim)
        self.bn2 = nn.BatchNorm1d(snn_dim)
        self.lif2 = snn.Leaky(beta=0.9, learn_beta=True)
        self.gru = nn.GRU(input_size + snn_dim, ann_dim, batch_first=True)
        self.head = nn.Linear(ann_dim, horizon)

    def forward(self, x, return_trajectory=False):
        B, T, _ = x.shape
        spk_enc, _, sum_spks = self.encoder(x)
        m2 = torch.zeros(B, self.snn_dim, device=x.device)
        mem_rec = []
        for t in range(spk_enc.size(0)):
            s2, m2 = self.lif2(self.bn2(spk_enc[t]), m2)
            mem_rec.append(m2); sum_spks += s2.detach().sum().item()
        
        m_rec = torch.stack(mem_rec)
        m_down = m_rec.permute(1,0,2).reshape(B, T, 4, self.snn_dim).mean(dim=2)
        h_comb = torch.cat([x, m_down], dim=-1)
        _, h_gru = self.gru(h_comb)
        preds = self.head(h_gru[-1])
        if return_trajectory: return preds, m_rec, sum_spks
        return preds, sum_spks

class SpikeGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.lif = snn.Leaky(beta=0.9)
    def forward(self, x, h_prev, m_prev):
        spk, m_next = self.lif(self.i2h(x) + self.h2h(h_prev), m_prev)
        return spk, m_next

class SNNForecasterV7(nn.Module):
    def __init__(self, input_size, horizon, hidden=256):
        super().__init__()
        self.hidden = hidden
        self.encoder = ConvSpikeEncoder(input_size, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.lif2 = snn.Leaky(beta=0.9)
        self.attn = SpikingAttentionSSA(hidden)
        self.head = nn.Linear(hidden, horizon)
    def forward(self, x, return_trajectory=False):
        spk1, _, sc = self.encoder(x)
        B = x.size(0)
        m2 = torch.zeros(B, self.hidden, device=x.device)
        mem_rec = []
        for t in range(spk1.size(0)):
            s2, m2 = self.lif2(self.bn2(spk1[t]), m2)
            mem_rec.append(m2); sc += s2.detach().sum().item()
        mr = torch.stack(mem_rec)
        feat, sca = self.attn(mr.permute(1,0,2))
        preds = self.head(feat)
        if return_trajectory: return preds, mr, sc+sca
        return preds, sc+sca

class HybridForecasterV7(nn.Module):
    def __init__(self, input_size, horizon, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.encoder = ConvSpikeEncoder(input_size, hidden)
        self.spike_gru = SpikeGRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, horizon)
    def forward(self, x, return_trajectory=False):
        spk1, _, sc = self.encoder(x)
        B = x.size(0)
        h = torch.zeros(B, self.hidden, device=x.device)
        m = torch.zeros(B, self.hidden, device=x.device)
        mem_rec = []
        for t in range(spk1.size(0)):
            spk_g, m = self.spike_gru(spk1[t], h, m)
            h = m # In Spike-GRU, the membrane potential acts as the hidden state
            mem_rec.append(h); sc += spk_g.detach().sum().item()
        preds = self.head(h)
        if return_trajectory: return preds, torch.stack(mem_rec), sc
        return preds, sc
