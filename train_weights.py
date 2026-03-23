import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

CLASSES = ["elephant", "mouse", "gpu_task", "interactive"]


def log_transform(X):
    """Log-scale features that span multiple orders of magnitude."""
    X_t = X.copy().astype(float)
    X_t[:, 0] = np.log1p(X_t[:, 0])   # byte_rate
    X_t[:, 1] = np.log1p(X_t[:, 1])   # pkt_rate
    X_t[:, 4] = np.log1p(X_t[:, 4])   # pkt_size
    return X_t


# ── Load data ─────────────────────────────────────────────────────
print("[*] Loading training data...")
X = np.load("train_X.npy")
y = np.load("train_y.npy")
print(f"[*] {len(X)} samples loaded")

# ── Log transform ─────────────────────────────────────────────────
X_log = log_transform(X)

# ── Normalise to [0, 1] ───────────────────────────────────────────
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_log)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# ── Train 6 → 32 → 4 MLP ─────────────────────────────────────────
print("[*] Training MLP...")
clf = MLPClassifier(
    hidden_layer_sizes=(32,),
    activation="relu",
    max_iter=500,
    random_state=42,
    alpha=0.01,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    verbose=False,
)
clf.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────
y_pred = clf.predict(X_test)
print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=CLASSES))

# ── Save weights and scaler ───────────────────────────────────────
os.makedirs("weights", exist_ok=True)

np.save("weights/W_ih.npy",         clf.coefs_[0])   # (6, 32)
np.save("weights/W_ho.npy",         clf.coefs_[1])   # (32, 4)
np.save("weights/scaler_min.npy",   scaler.data_min_)
np.save("weights/scaler_range.npy", scaler.data_range_)

print(f"\n[*] Weights saved to weights/")
