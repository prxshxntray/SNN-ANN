import numpy as np
import os

rng = np.random.default_rng(42)
X, y = [], []
N = 500

for _ in range(N):
    # Elephant: high byte rate, large packets, LOW burst
    X.append([
        rng.uniform(5e7, 1e9),      # byte_rate   HIGH
        rng.uniform(100, 500),      # pkt_rate    MEDIUM
        rng.uniform(0.002, 0.02),   # mean_iat    MEDIUM
        rng.uniform(0.001, 0.008),  # std_iat     LOW
        rng.uniform(1100, 1500),    # pkt_size    LARGE
        rng.uniform(0.0, 0.15),     # burst_ratio LOW
    ])
    y.append(0)

    # Mouse: LOW byte rate, small packets, HIGH burst
    X.append([
        rng.uniform(100, 80000),    # byte_rate   LOW
        rng.uniform(1, 40),         # pkt_rate    LOW
        rng.uniform(0.05, 0.8),     # mean_iat    HIGH
        rng.uniform(0.02, 0.3),     # std_iat     HIGH
        rng.uniform(40, 250),       # pkt_size    SMALL
        rng.uniform(0.6, 1.0),      # burst_ratio HIGH
    ])
    y.append(1)

    # GPU task: VERY high byte rate, low iat, HIGH burst
    X.append([
        rng.uniform(4e8, 1e9),      # byte_rate   VERY HIGH
        rng.uniform(600, 2000),     # pkt_rate    VERY HIGH
        rng.uniform(0.0001, 0.001), # mean_iat    VERY LOW
        rng.uniform(0.0001, 0.001), # std_iat     VERY LOW
        rng.uniform(1000, 1500),    # pkt_size    LARGE
        rng.uniform(0.7, 1.0),      # burst_ratio VERY HIGH
    ])
    y.append(2)

    # Interactive: must cover very small values
    X.append([
        rng.uniform(100,   50000),   # byte_rate  ← lower min from 500
        rng.uniform(5,     150),     # pkt_rate   ← lower min from 10
        rng.uniform(0.0005, 0.008),  # mean_iat   ← lower min from 0.001
        rng.uniform(0.0,   0.003),   # std_iat    unchanged
        rng.uniform(40,    180),     # pkt_size   unchanged
        rng.uniform(0.0,   0.08),    # burst_ratio unchanged
    ])
    y.append(3)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
X = np.clip(X, 0, None)

idx = rng.permutation(len(y))
X, y = X[idx], y[idx]

os.makedirs("weights", exist_ok=True)
np.save("train_X.npy", X)
np.save("train_y.npy", y)

print(f"Generated {len(X)} samples")
for i, name in enumerate(["elephant","mouse","gpu_task","interactive"]):
    print(f"  {name:<12}: {int((y==i).sum())} samples")
