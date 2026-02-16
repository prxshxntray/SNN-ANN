# N-MNIST Neuromorphic Vision Benchmark 🧠⚡️

## Overview
This repository contains a comprehensive benchmark analysis comparing diverse neural network architectures on the **N-MNIST (Neuromorphic MNIST)** dataset. The study evaluates the trade-offs between biological plausibility, energy efficiency, and classification speed.

## Models Compared
1.  **SNN (Spiking Neural Network)**: Fully spiking CNN with rate-coded output.
2.  **RNN (Recurrent Neural Network)**: GRU-based architecture processing flattened event frames.
3.  **Hybrid SNN-ANN**: A novel architecture combining a spiking feature extractor with an ANN classifier for rapid decision-making.
4.  **CNN Baseline**: Standard CNN trained on accumulated event frames (static baseline).

## Key Findings 🏆
| Model | Accuracy | Decision Latency (Steps) | Avg Spikes / Energy Proxy |
| :--- | :--- | :--- | :--- |
| **Hybrid SNN-ANN** | **98.71%** | **2.0** (Fastest) | ~16,241 |
| **SNN** | 98.50% | 17.0 (Slowest) | ~59,774 |
| **RNN (GRU)** | 98.39% | 4.0 | 125M MACs |
| **CNN Baseline** | 95.66% | 7.0 | 6.7M MACs |

### 🚀 The "Hybrid Advantage"
Our results demonstrate that the **Hybrid SNN-ANN** architecture achieves the best balance of performance, reaching the highest accuracy with the fastest time-to-decision (just 2 time steps). This confirms the efficacy of **mixed-signal approaches** for real-world neuromorphic vision tasks.

## Repository Contents
-   `NMNIST_SNN_Analysis.ipynb`: The main Jupyter Notebook containing:
    -   Data preprocessing with `tonic` (event-to-frame conversion).
    -   Model definitions (SNN, RNN, Hybrid, CNN).
    -   Training loops and evaluation logic.
    -   **Research-Grade Metrics**: Latency analysis, Confidence-over-Time plots, and Energy estimation.

## Getting Started
1.  **Install Dependencies**:
    Requires `torch`, `torchvision`, `spikingjelly`, `tonic`, `matplotlib`, `seaborn`, `pandas`, `thop`.
    ```bash
    pip install torch torchvision spikingjelly tonic matplotlib seaborn pandas thop
    ```
2.  **Run the Analysis**:
    Open `NMNIST_SNN_Analysis.ipynb` in Jupyter Lab or Notebook.
    The notebook will automatically download the N-MNIST dataset (cached via `tonic`) and run the full benchmark pipeline.

## License
MIT License.
