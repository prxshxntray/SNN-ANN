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
