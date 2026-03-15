# SNN-ANN: Architecture Evolution Log

This document tracks the technical development and design decisions of the SNN-ANN project, focusing on the PJM Wind Forecasting models.

## Phase 1: V4 Baseline (Foundational SNN)
The initial implementation focused on basic spiking dynamics for time-series forecasting.

- **Architecture**: Simple 2-layer SNN with discrete spikes.
- **Input**: Raw time-series values.
- **Output**: Linear readout of the final hidden layer spikes.
- **Challenges**: Difficulty in representing continuous values and instability in gradient flow.

## Phase 2: V5 - SNNTorch Integration & Stability
Significant upgrades were made to leverage the `snnTorch` library for more sophisticated dynamics.

### 1. Advanced Neuron Dynamics (PLIF)
- **Transition**: Moved from basic LIF to **Parametric LIF (PLIF)** neurons.
- **Innovation**: Enabled learnable `beta` (leak) parameters, allowing neurons to adapt their temporal resolution to the data.
- **Safety**: Implemented **stability clamps** (0.0–0.99) on `beta` to prevent exponential membrane potential explosion.

### 2. Input/Output Refinement
- **Raw + Delta Input**: Modernized the input pipeline to feed both absolute values and temporal changes, helping the SNN detect "shocks" in wind speed.
- **Continuous Readout**: Abandoned discrete spike outputs for the final layer in favor of **Membrane Potential Readout**, enabling true continuous regression.

### 3. SNN-GRU Hybrid Paradigms
- **V4 Hybrid**: Parallel residual design where SNN and GRU processed the same data independently.
- **V5 Hybrid (Chained)**: Shifted to a **Feature Extraction** paradigm.
    - SNN processes deltas to extract rich temporal features.
    - GRU consumes these features alongside raw data for final forecasting.

## Phase 3: Research-Grade Enhancements
Introduced better diagnostics and training stability.

- **Temporal Attention**: Integrated a learned attention mechanism over the spike sequence.
- **Adaptive Surrogate Gradients**: Implemented dynamic slope sharpening during training to ensure active spiking and clean gradients.
- **Energy Diagnostics**: Added native spike-counting methods to models for future neuromorphic efficiency comparisons.

---

## Phase 4: Literature Review — Techniques for Next-Generation Improvements

Eight research papers were reviewed to identify actionable techniques for improving the current SNN/Hybrid model. Below is a per-paper summary of extractions and their intended impact.

---

### Paper 1: SpikSTAG — Spiking Spatial-Temporal Adaptive Graph Neural Network
**Source**: SpikSTAG_GNN-SNN.pdf

**Key Techniques Extracted**:
- **Dual-Path Spike Fusion (DSF)**: Combines an LSTM path and a Spiking Self-Attention (SSA) path via a learned gating mechanism. The LSTM captures sequential dependencies while SSA captures long-range correlations—fused dynamically.
- **Multi-Scale Spike Aggregation (MSSA)**: Extracts spatial features at multiple scales using spike-based operations, reducing computational cost vs. dense GNNs.
- **Adaptive Graph Learning**: Learns spatial relationships from data rather than requiring predefined graphs, using node embeddings with softmax normalization.
- **Spiking Self-Attention (SSA)**: Replaces softmax attention with spike-based binary masking (Q·K^T), eliminating floating-point multiplications.

**How It Helps Our Model**:
- **Dual-path fusion** maps directly to our Hybrid paradigm—we can add a spiking attention path alongside our GRU to dynamically gate feature contributions.
- **SSA** can replace our current scalar attention with a hardware-friendly spike-based version.
- **Energy**: Reported 70-90% energy savings vs. ANN counterparts.

---

### Paper 2: Spikeformer — Training High-Performance SNNs with Transformers
**Source**: traininghighperformSNN.pdf

**Key Techniques Extracted**:
- **Convolutional Tokenizer (CT)**: Replaces naive patch embedding with convolution layers to add local inductive bias, stabilizing training and reducing data hunger.
- **Spatio-Temporal Attention (STA)**: Extends vanilla self-attention to jointly capture spatial and temporal features across spike timesteps—key for event patterns.
- **LIF-based Q/K/V**: All attention queries, keys, and values pass through spiking neurons, keeping the entire computation spike-based.

**How It Helps Our Model**:
- **CT module** can be used as a front-end encoder to convert raw time-series patches into stable spike representations before the SNN layers.
- **STA** provides a principled upgrade over our simple temporal attention, jointly modeling spatial (across features) and temporal (across timesteps) correlations.
- Used `snnTorch`-compatible LIF neurons, directly applicable to our codebase.

---

### Paper 3: PWM-Based SNN for Univariate Time-Series Forecasting
**Source**: SNNforUnivariateTimeSeriesForecasting.pdf (Lucas & Portillo, 2024)

**Key Techniques Extracted**:
- **PulseWidth Modulation (PWM) Encoding**: Temporal encoding that converts continuous values to precise spike timings via PWM carrier signals. Preserves more information than rate coding while maintaining low latency (single time-step).
- **PWM-Based Loss Function**: Custom loss computed on spike timing differences (δ_output − δ_target)², directly capturing temporal prediction accuracy rather than generic MSE on membrane potentials.
- **First-Spike Decoding**: Only the first spike per carrier is decoded, reducing noise from multi-spike outputs.
- **Ultra-Low Latency**: Single time-step solution—no need for long spike windows.
- **Robust to Weight Init**: Results don't vary significantly with different weight initializations.

**How It Helps Our Model**:
- **PWM encoding** could replace our raw+delta input, providing a more precise spike representation that preserves continuous amplitude information in spike timing.
- **Spike-timing loss** offers a fundamentally different training signal vs. our current MSE on membrane potentials—could help the SNN learn finer temporal dynamics.
- **Single-step inference** would dramatically improve inference speed on neuromorphic hardware.

---

### Paper 4: Bespoke Loss Functions for SNN Time-Series Forecasting
**Source**: BespokeLossFunctionsForSNN.pdf (Manna et al., 2024)

**Key Techniques Extracted**:
- **Derivative Spike Encoding**: Inspired by event cameras—encodes the *derivative* (rate of change) of the signal rather than its absolute value. Uses multiple threshold neurons for different magnitudes of change. Simultaneously applies a differencing transform, increasing stationarity.
- **DecodingLoss**: Reconstructs the signal from output spikes using cumulative sums weighted by neuron thresholds, then computes MSE in the *original signal domain*. Consistently outperforms SLAYER's SpikeTime loss.
- **ISILoss**: Heuristic based on inter-spike intervals using cumulative sums and time-step rescaling. Less effective than DecodingLoss but provides alternative gradient signals.
- **SLAYER Learning Rule**: Uses surrogate gradients with weight AND delay learning via FIR exponential kernel convolution.

**How It Helps Our Model**:
- **Derivative encoding** aligns perfectly with our "delta" input concept but formalizes it with multi-threshold neurons for granularity control.
- **DecodingLoss** is the most impactful technique—by optimizing in the reconstructed signal domain, the SNN has more freedom in how it represents values, expanding the loss landscape and improving convergence.
- **Threshold optimization** insights: more neurons ≠ better; careful threshold selection based on data distribution analysis is critical.

---

### Paper 5: Derivative Spike Encoding & SNNs (Extended Version)
**Source**: DerivativeSpikeEncoding&SNNs.pdf (Manna et al., HICSS 2025)

**Key Additional Insights** (extends Paper 4):
- **SARIMA Comparison**: DecodingLoss SNN achieved MSE of 2531 vs. SARIMA's 4948 on Panama dataset — ~48% improvement.
- **Emphasis on stationarity**: Differencing transform embedded in encoding increases stationarity, making forecasting more reliable.
- **Second-derivative exposure**: The derivative encoding implicitly exposes information about the second derivative (inflection points), enabling richer temporal learning.

**How It Helps Our Model**:
- Validates that the derivative approach + DecodingLoss can beat classical methods, supporting its adoption in our pipeline.
- Stationarity insight suggests we should consider formal differencing of our PJM wind data before encoding.

---

### Paper 6: Efficient & Effective Time-Series Forecasting with SNNs (SeqSNN)
**Source**: Effecient&EffectiveTimeSeriesForecastingwithSNN.pdf (Lv et al., ICML 2024)

**Key Techniques Extracted**:
- **Temporal Alignment Framework**: Bridges time-series timestep ΔT and SNN timestep Δt via Tₛ spiking sub-steps per data step (ΔT = Tₛ·Δt). Critical for proper temporal granularity.
- **Delta Spike Encoder**: `S = SN(BN(Linear(xₜ − xₜ₋₁)))` — learnable linear projection of temporal differences followed by batch norm and spiking neurons.
- **Convolutional Spike Encoder**: `S = SN(BN(Conv(X)))` — captures shape/morphological features in subsequences. **Outperforms delta encoder** by average R² improvement of 0.09.
- **Spike-RNN / Spike-GRU**: SNN counterparts that *persist* membrane potential across time steps. **Spike-RNN significantly outperforms vanilla GRU** and achieves SOTA on several benchmarks.
- **iSpikformer**: Spiking version of iTransformer with SSA. Achieves comparable results to iTransformer with ~66-75% energy reduction.
- **Decay Rate (β) Sensitivity**: Higher β (0.95-0.99) yields better R², as SNNs retain more long-term temporal state.
- **Energy Reduction**: Average ~70.33% energy reduction over ANN counterparts on 45nm neuromorphic hardware.
- **Repetition encoding fails**: Simply repeating values as spike input causes convergence failure—proper temporal encoding is essential.

**How It Helps Our Model**:
- **Most directly applicable paper**. Our GRU → Spike-GRU migration is strongly supported.
- **Conv spike encoder** should replace our raw+delta input — it captures morphological features and significantly outperforms delta encoding alone.
- **Temporal alignment (Tₛ)** is a critical missing hyperparameter in our architecture—we need to define spiking sub-steps per data timestep.
- **β=0.95-0.99** validates our current clamped range—we should default toward 0.95.
- **Batch normalization before spiking** is a key stabilization technique we're not using.

---

### Paper 7: Reinforcement Learning through Modulation of STDP
**Source**: RL modulation of STDPP.pdf (Florian, 2007)

**Key Techniques Extracted**:
- **Modulated STDP (MSTDP)**: Standard STDP modulated by reward signal: `dw/dt = γ·r(t)·ξ(t)`. Hebbian STDP when reward is positive, anti-Hebbian when negative.
- **MSTDP with Eligibility Trace (MSTDPET)**: Adds an exponentially decaying eligibility trace `z(t)` at each synapse: `dw/dt = γ·r(t)·z(t)`. Enables learning even with delayed rewards.
- **Analytical Justification**: Derived from OLPOMDP applied to Spike Response Model — provides theoretical convergence guarantees.
- **Bidirectional Associative STDP**: With homeostatic threshold adaptation, both LTP and LTD become associative and spike-timing-dependent.
- **Spike-timing-dependent intrinsic plasticity**: Firing thresholds can also be adapted using RL, complementing synaptic plasticity.

**How It Helps Our Model**:
- **Eligibility traces** provide a biological mechanism for credit assignment over time—directly relevant for forecasting where reward (loss) is computed at the end of a sequence.
- **R-STDP** could be used as an *auxiliary learning signal* alongside backprop, fine-tuning synaptic connections based on forecasting accuracy as reward.
- **Intrinsic plasticity** (adaptive firing thresholds) is a more biologically grounded alternative to our learnable β approach.
- **Long-term**: Foundation for eventual neuromorphic deployment with on-chip learning.

---

### Paper 8: Meta-Learning in SNNs with Reward-Modulated STDP
**Source**: Meta-Learning SNNs .pdf (Khoee et al., 2023)

**Key Techniques Extracted**:
- **Three-Component Architecture**: Convolutional layer (STDP) → Memory layer (R-STDP with stochastic LIF) → Decision layer (adaptive R-STDP). Mirrors hippocampus + PFC + VTA.
- **Episodic Memory with Stochastic LIF Neurons**: Memory layer uses LIF neurons with stochastic thresholds (ρ = ρ_θ · exp((u-u_θ)/Δu)) to encode past experiences and prevent catastrophic forgetting.
- **Sparsity-Controlled Memory**: Reward/punishment intervals control that only ~15% of neurons activate per sample (c=15%, s=3%), preventing overwriting.
- **Adaptive R-STDP**: Reward/punishment values dynamically adjust based on classification accuracy—starts at ±0.5 and updates each task.
- **Lateral Inhibition**: Between neuron groups for competitive dynamics and faster convergence.
- **Results**: 99.06% on Omniglot 5-way 1-shot, competitive with MAML and SNAIL despite being fully bio-plausible.

**How It Helps Our Model**:
- **Stochastic LIF neurons** could improve robustness of our SNN to noisy wind data by adding controlled randomness to spike generation.
- **Sparsity control via R-STDP** offers a principled approach to prevent the "dead neuron" problem we see in training.
- **Adaptive reward scaling** can be adapted for our regression task—scale gradients based on recent forecasting accuracy.
- **Lateral inhibition** can increase feature diversity in our SNN layers.
- **Memory mechanism** could enable continual/online learning for adapting to seasonal patterns.

---

## Proposed Improvement Roadmap

Based on the literature review, improvements are prioritized by expected impact and implementation effort:

### Tier 1: High Impact, Moderate Effort
1. **Convolutional Spike Encoder** (Paper 6) — Replace raw+delta with learned conv encoder + BN
2. **DecodingLoss / Spike-Timing Loss** (Papers 3, 4) — Add domain-aware loss alongside MSE
3. **Temporal Alignment (Tₛ sub-steps)** (Paper 6) — Add configurable spiking sub-steps per timestep
4. **Batch Normalization before Spiking** (Paper 6) — Add BN layers before all SN layers

### Tier 2: High Impact, Higher Effort
5. **Spiking Self-Attention (SSA)** (Papers 1, 2) — Replace scalar attention with spike-based Q·K^T
6. **Spike-GRU/RNN Migration** (Paper 6) — Convert GRU to spike-based version with membrane persistence
7. **Dual-Path Fusion** (Paper 1) — Parallel SNN + attention paths with gated combination

### Tier 3: Research-Grade Extensions
8. **Derivative Encoding with Stationarity** (Papers 4, 5) — Formal differencing + multi-threshold encoding
9. **R-STDP Auxiliary Learning** (Papers 7, 8) — Reward-modulated fine-tuning signal
10. **Stochastic LIF + Sparsity Control** (Paper 8) — Noise robustness + dead neuron prevention
11. **Adaptive Reward Scaling** (Paper 8) — Dynamic gradient scaling based on recent performance
