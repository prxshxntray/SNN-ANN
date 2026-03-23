import numpy as np
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor,
    PoissonGroup, run, ms, Hz,
    prefs, start_scope, defaultclock
)

prefs.codegen.target = "numpy"

CLASSES = ["elephant", "mouse", "gpu_task", "interactive"]


class SNNFlowClassifier:
    """
    Hybrid SNN classifier:
    - Input → Hidden: genuine Brian2 LIF spiking
    - Hidden spike counts → Output: linear readout using trained W_ho
    This mirrors real neuromorphic deployment (e.g. Intel Loihi readout layer).
    """

    SIM_DURATION = 500
    MAX_RATE_HZ  = 200
    MIN_RATE_HZ  = 10

    def __init__(self, weights_dir="weights"):
        print("[*] Loading trained weights into Brian2 SNN...")

        self.W_ih_orig = np.load(f"{weights_dir}/W_ih.npy")   # (6, 32)
        self.W_ho_orig = np.load(f"{weights_dir}/W_ho.npy")   # (32, 4)
        self.scaler_min   = np.load(f"{weights_dir}/scaler_min.npy")
        self.scaler_range = np.load(f"{weights_dir}/scaler_range.npy")

        # Scale only W_ih for LIF dynamics
        # W_ho stays original — used as linear readout decoder
        self.W_ih = self.W_ih_orig / (np.abs(self.W_ih_orig).max() + 1e-9) * 2.0

        print(f"    W_ih: {self.W_ih.shape}  W_ho: {self.W_ho_orig.shape}")

    def _normalise(self, features: np.ndarray) -> np.ndarray:
        f = features.copy().astype(float)
        f[0] = np.log1p(f[0])
        f[1] = np.log1p(f[1])
        f[4] = np.log1p(f[4])
        norm = (f - self.scaler_min) / (self.scaler_range + 1e-9)
        return np.clip(norm, 0, 1)

    def _run_hidden_layer(self, rates_hz: np.ndarray) -> np.ndarray:
        """
        Run Brian2 LIF hidden layer.
        Returns normalised spike rate per hidden neuron (32,).
        """
        start_scope()
        defaultclock.dt = 0.1 * ms

        lif_eqs = """
            dv/dt = (v_rest - v + I) / tau : 1 (unless refractory)
            dI/dt = -I / tau_s             : 1
            v_rest : 1 (constant)
            tau    : second (constant)
            tau_s  : second (constant)
        """

        # Input Poisson group
        input_layer = PoissonGroup(6, rates=rates_hz * Hz)

        # Hidden LIF layer
        hidden_layer = NeuronGroup(
            32, lif_eqs,
            threshold="v > 1.0",
            reset="v = 0",
            refractory=2 * ms,
            method="euler",
        )
        hidden_layer.v      = 0
        hidden_layer.I      = 0
        hidden_layer.v_rest = 0
        hidden_layer.tau    = 10 * ms
        hidden_layer.tau_s  = 5  * ms

        # Input → Hidden synapses
        syn_ih = Synapses(
            input_layer, hidden_layer,
            "w : 1",
            on_pre="I_post += w",
        )
        syn_ih.connect()
        for i in range(6):
            for j in range(32):
                syn_ih.w[i * 32 + j] = float(self.W_ih[i, j])

        # Monitor hidden spikes
        hidden_mon = SpikeMonitor(hidden_layer)

        run(self.SIM_DURATION * ms)

        # Count spikes per hidden neuron
        spike_counts = np.zeros(32)
        for idx in hidden_mon.i:
            spike_counts[idx] += 1

        # Normalise by simulation duration to get firing rates
        spike_rates = spike_counts / (self.SIM_DURATION / 1000.0)

        return spike_rates

    def classify(self, features: np.ndarray) -> str:
        """
        1. Normalise features
        2. Rate-encode to Poisson inputs
        3. Run Brian2 LIF hidden layer → hidden spike rates
        4. Linear readout: hidden_rates @ W_ho → class scores
        5. Winner-takes-all
        """
        norm     = self._normalise(features)
        rates_hz = self.MIN_RATE_HZ + norm * (self.MAX_RATE_HZ - self.MIN_RATE_HZ)

        # Step 3 — Brian2 hidden layer
        hidden_rates = self._run_hidden_layer(rates_hz)

        # Step 4 — if hidden layer silent, fall back to pure MLP
        if hidden_rates.sum() == 0:
            hidden = np.maximum(0, norm @ self.W_ih_orig)
            scores = hidden @ self.W_ho_orig
            return CLASSES[int(np.argmax(scores))]

        # Step 5 — linear readout using original W_ho
        scores = hidden_rates @ self.W_ho_orig
        return CLASSES[int(np.argmax(scores))]


if __name__ == "__main__":
    clf = SNNFlowClassifier()

    tests = {
        "elephant":    np.array([1e8,  100,  0.01,  0.002, 1400, 0.05]),
        "mouse":       np.array([1e4,   20,  0.05,  0.02,   200, 0.80]),
        "gpu_task":    np.array([8e8,  900,  0.001, 0.001, 1400, 0.90]),
        "interactive": np.array([1e4,   50,  0.003, 0.001,  100, 0.04]),
    }

    print("\n── Brian2 SNN Smoke Test ──")
    passed = 0
    for expected, feat in tests.items():
        result = clf.classify(feat)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        print(f"  {status}  Expected: {expected:<12}  Got: {result}")

    print(f"\n{passed}/4 tests passed")
