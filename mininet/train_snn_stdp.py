"""
Hybrid SNN training pipeline.
Step 1: Pretrain with MLP to get good initial weights.
Step 2: Fine-tune with reward-modulated STDP using Brian2 spiking dynamics.
"""

import numpy as np
import os
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor,
    PoissonGroup, run, ms, Hz,
    prefs, start_scope, defaultclock
)
from sklearn.neural_network import MLPClassifier

prefs.codegen.target = "numpy"

CLASSES      = ["elephant", "mouse", "gpu_task", "interactive"]
N_INPUT      = 6
N_HIDDEN     = 32
N_OUTPUT     = 4
SIM_DURATION = 100   # ms per sample
MAX_RATE     = 200   # Hz
MIN_RATE     = 10    # Hz
N_EPOCHS     = 5


# ── Preprocessing ─────────────────────────────────────────────────

def log_transform(X):
    X_t = X.copy().astype(float)
    X_t[:, 0] = np.log1p(X_t[:, 0])
    X_t[:, 1] = np.log1p(X_t[:, 1])
    X_t[:, 4] = np.log1p(X_t[:, 4])
    return X_t


def compute_scaler(X_log):
    smin   = X_log.min(axis=0)
    srange = X_log.max(axis=0) - X_log.min(axis=0)
    return smin, srange


def normalise(X, smin, srange):
    return np.clip((X - smin) / (srange + 1e-9), 0, 1)


# ── Single sample forward + STDP update ───────────────────────────

def run_sample(norm_feat, W_ih, W_ho,
               train=False, true_label=None,
               lr_pos=0.005, lr_neg=0.002):
    """
    Run one sample through the SNN.
    - Hidden layer: genuine Brian2 LIF spiking
    - Output: linear readout on hidden spike rates
    - If train=True: apply reward-modulated STDP update
    Returns: pred, hid_rates, out_counts, W_ih, W_ho
    """
    start_scope()
    defaultclock.dt = 0.1 * ms

    rates_hz = MIN_RATE + norm_feat * (MAX_RATE - MIN_RATE)

    lif_eqs = """
        dv/dt = (v_rest - v + I) / tau : 1 (unless refractory)
        dI/dt = -I / tau_s             : 1
        v_rest : 1 (constant)
        tau    : second (constant)
        tau_s  : second (constant)
    """

    # ── Input ─────────────────────────────────────────────────────
    inp = PoissonGroup(N_INPUT, rates=rates_hz * Hz)

    # ── Hidden ────────────────────────────────────────────────────
    hid = NeuronGroup(
        N_HIDDEN, lif_eqs,
        threshold="v > 1.0",
        reset="v = 0",
        refractory=2 * ms,
        method="euler",
    )
    hid.v=0; hid.I=0; hid.v_rest=0
    hid.tau=10*ms; hid.tau_s=5*ms

    # ── Synapses: Input → Hidden ───────────────────────────────────
    s_ih = Synapses(inp, hid, "w:1", on_pre="I_post+=w")
    s_ih.connect()
    for i in range(N_INPUT):
        for j in range(N_HIDDEN):
            s_ih.w[i * N_HIDDEN + j] = float(W_ih[i, j])

    # ── Monitor ───────────────────────────────────────────────────
    hid_mon = SpikeMonitor(hid)

    run(SIM_DURATION * ms)

    # Hidden spike rates (Hz)
    hid_counts = np.zeros(N_HIDDEN)
    for idx in hid_mon.i:
        hid_counts[idx] += 1
    hid_rates = hid_counts / (SIM_DURATION / 1000.0)

    # Linear readout using W_ho
    scores = hid_rates @ W_ho
    pred   = int(np.argmax(scores))

    # ── STDP update ───────────────────────────────────────────────
    if train and true_label is not None:
        norm_rates = hid_rates / (hid_rates.max() + 1e-9)
        input_rates = rates_hz / MAX_RATE

        # Update W_ho: potentiate correct class, depress others
        for j in range(N_OUTPUT):
            if j == true_label:
                W_ho[:, j] += lr_pos * norm_rates * (1 - W_ho[:, j])
            else:
                W_ho[:, j] -= lr_neg * norm_rates * (W_ho[:, j] + 1)

        # Update W_ih based on prediction correctness
        if pred == true_label:
            W_ih += lr_pos * np.outer(
                input_rates,
                norm_rates,
            )
        else:
            W_ih -= lr_neg * np.outer(
                input_rates,
                norm_rates,
            )

        # Clip to prevent explosion
        W_ih = np.clip(W_ih, -3.0, 3.0)
        W_ho = np.clip(W_ho, -3.0, 3.0)

    return pred, hid_rates, hid_counts, W_ih, W_ho


# ── Main training function ─────────────────────────────────────────

def train():
    # ── Load data ─────────────────────────────────────────────────
    print("[*] Loading data...")
    X_raw = np.load("train_X.npy")
    y     = np.load("train_y.npy")

    X_log        = log_transform(X_raw)
    smin, srange = compute_scaler(X_log)
    X_norm       = normalise(X_log, smin, srange)

    print(f"[*] {len(X_raw)} samples, {N_EPOCHS} epochs")

    # ── Step 1: MLP pretraining ───────────────────────────────────
    print("\n[*] Step 1: MLP pretraining...")
    clf = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="relu",
        max_iter=500,
        random_state=42,
        alpha=0.01,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
    )
    clf.fit(X_norm, y)
    mlp_acc = clf.score(X_norm, y)
    print(f"[*] MLP pretrain accuracy: {mlp_acc*100:.1f}%")

    # Use MLP weights as STDP starting point
    W_ih = clf.coefs_[0].copy()    # (6, 32)
    W_ho = clf.coefs_[1].copy()    # (32, 4)

    # Scale W_ih for LIF dynamics, keep W_ho as linear readout
    W_ih = W_ih / (np.abs(W_ih).max() + 1e-9) * 2.0

    # ── Step 2: STDP fine-tuning ──────────────────────────────────
    print("\n[*] Step 2: STDP fine-tuning...")
    rng = np.random.default_rng(42)

    for epoch in range(N_EPOCHS):
        idx     = rng.permutation(len(y))
        correct = 0
        lr_pos  = 0.005 * (0.8 ** epoch)
        lr_neg  = 0.002 * (0.8 ** epoch)

        for n, i in enumerate(idx):
            pred, _, _, W_ih, W_ho = run_sample(
                X_norm[i], W_ih, W_ho,
                train=True,
                true_label=int(y[i]),
                lr_pos=lr_pos,
                lr_neg=lr_neg,
            )
            if pred == y[i]:
                correct += 1

            if (n + 1) % 200 == 0:
                acc = correct / (n + 1) * 100
                print(f"  Epoch {epoch+1}/{N_EPOCHS}  "
                      f"sample {n+1}/{len(y)}  "
                      f"acc={acc:.1f}%  "
                      f"lr={lr_pos:.5f}")

        epoch_acc = correct / len(y) * 100
        print(f"── Epoch {epoch+1} complete: "
              f"accuracy={epoch_acc:.1f}%")

    # ── Save weights ──────────────────────────────────────────────
    os.makedirs("weights", exist_ok=True)
    np.save("weights/W_ih.npy",         W_ih)
    np.save("weights/W_ho.npy",         W_ho)
    np.save("weights/scaler_min.npy",   smin)
    np.save("weights/scaler_range.npy", srange)
    print("\n[*] Weights saved to weights/")

    # ── Final evaluation ──────────────────────────────────────────
    print("\n── Final Evaluation (200 samples) ──")
    correct       = 0
    class_correct = np.zeros(N_OUTPUT)
    class_total   = np.zeros(N_OUTPUT)

    eval_idx = rng.choice(len(y), size=min(200, len(y)), replace=False)
    for i in eval_idx:
        pred, _, _, _, _ = run_sample(X_norm[i], W_ih, W_ho)
        if pred == y[i]:
            correct += 1
            class_correct[y[i]] += 1
        class_total[y[i]] += 1

    print(f"Overall accuracy: {correct/len(eval_idx)*100:.1f}%")
    for i, name in enumerate(CLASSES):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i] * 100
            print(f"  {name:<12}: {acc:.1f}%")

    # ── Smoke test ────────────────────────────────────────────────
    print("\n── Smoke Test ──")
    tests = {
        "elephant":    np.array([1e8,  100,  0.01,  0.002, 1400, 0.05]),
        "mouse":       np.array([1e4,   20,  0.05,  0.02,   200, 0.80]),
        "gpu_task":    np.array([8e8,  900,  0.001, 0.001, 1400, 0.90]),
        "interactive": np.array([1e4,   50,  0.003, 0.001,  100, 0.04]),
    }
    for expected, feat in tests.items():
        f = feat.copy().astype(float)
        f[0] = np.log1p(f[0])
        f[1] = np.log1p(f[1])
        f[4] = np.log1p(f[4])
        norm_f = np.clip((f - smin) / (srange + 1e-9), 0, 1)
        pred, _, _, _, _ = run_sample(norm_f, W_ih, W_ho)
        status = "✓" if CLASSES[pred] == expected else "✗"
        print(f"  {status}  Expected: {expected:<12}  Got: {CLASSES[pred]}")


if __name__ == "__main__":
    train()
