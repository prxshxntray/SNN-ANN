# SNN-ANN

**Spiking Neural Networks × Artificial Neural Networks**  
**Hybrid architectures for energy-efficient intelligence systems**

This repository explores SNNs and hybrid SNN-ANN architectures with an applied lens on energy-efficient inference and infrastructure-scale optimisation. Long-term, this work supports the larger goal of building sustainable intelligence layers (e.g., WATTR) where efficiency matters as much as performance.

---

## Repository Structure

- `benchmarking/` — state-of-the-art understanding and reproduction
- `randomDatasets/` — experiments on public datasets
- `wattrDatasets/` — future private datasets (currently placeholder)

---

## What’s in Each Folder

### `benchmarking/` — Research backbone

**Purpose:** Understand the current state-of-the-art in SNNs and hybrid architectures.

Typical focus:
- Reproducing papers
- ANN vs SNN comparisons
- Latency/sparsity/energy proxies
- Hybrid conversion pipelines
- Training stability and surrogate gradients

**Goal:** Scientifically grounded clarity: where do these models help, and where do they fail?

---

### `randomDatasets/` — Applied experiments on public data

**Purpose:** Test SNN/hybrid feasibility on accessible datasets.

You might find:
- Vision experiments (N-MNIST)
- Time-series experiments (PJM Wind Forecasting)
- Benchmark ML workloads

**Objective:** Identify dataset characteristics that favour event-driven computation and hybrid gating approaches.

**Architectural Evolution (PJM Time-Series Forecasting):**
- **V1 (Initial):** Basic SNN rate encoding and parallel hybrid gating. *Result: SNN failed to learn continuous values (MAE ~3395 MW vs GRU ~368 MW).*
- **V2 (Delta + Deep LIF):** Shifted to Delta encoding (tracking changes), 128h 3-layer LIF, and temporal attention. *Result: SNN spiked properly but remained blind to absolute power levels.*
- **V3 (Raw+Delta & Residual Hybrid):** SNN receives concatenated Raw+Delta state. Hybrid model uses a **Residual Design** (zero-initialized gate adding SNN 'shock detection' features to a robust GRU baseline). *Result: Stable learning merging ANN baseline safety with SNN event-detection.*

---

### `wattrDatasets/` — Applied, future private data

**Purpose:** Design the pipeline for private datasets tied to infrastructure.

Reserved for (future):
- Data-centre telemetry
- Thermal/load signals
- Workload–cooling interactions
- Optimisation task framing

**Note:** This folder documents the intended structure + schema assumptions, but will remain private once data exists.

---

## Why SNN-ANN Hybrids?

SNNs offer:
- Event-driven computation
- Temporal sensitivity
- Potential efficiency benefits in the right regime

ANNs offer:
- Stable training
- Mature tooling
- Strong baselines

**Hybrid architectures** aim to combine stability with sparsity-driven behaviour.

> Core question: where do hybrids produce real, measurable advantages?

---

## Long-Term Vision

This research is aligned with building efficient intelligence systems for physical infrastructure—where energy/water matters and deployment constraints are real (edge, latency, cost, safety).

---

## Next Up (Roadmap)

- Benchmark suite consistency across tasks
- Better reporting: metrics dashboards / tables
- Baseline repos: standard ANN baselines for every experiment
- Optional: add a minimal CLI (`python -m ...`) for reproducibility

---

## License / Contributions

TBD.

---
