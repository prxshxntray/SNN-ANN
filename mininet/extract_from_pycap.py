# extract_from_pcap.py
"""
Reads training_capture.pcap and extracts per-flow features + labels.
Label assignment is based on port number (matches traffic_gen.py):
  5101 → elephant
  5102 → mouse
  5103 → gpu_task
  5104 → interactive
"""

import numpy as np
from scapy.all import rdpcap, IP, TCP, UDP
from collections import defaultdict

PORT_TO_LABEL = {
    5101: 0,   # elephant
    5102: 1,   # mouse
    5103: 2,   # gpu_task
    5104: 3,   # interactive
}
LABEL_NAMES = ["elephant", "mouse", "gpu_task", "interactive"]


def extract_features_from_pcap(pcap_path: str):
    print(f"[*] Reading {pcap_path}...")
    packets = rdpcap(pcap_path)
    print(f"[*] {len(packets)} packets loaded")

    # Group packets by flow (5-tuple)
    flows = defaultdict(lambda: {"times": [], "sizes": [], "label": None})

    for pkt in packets:
        if not pkt.haslayer(IP):
            continue

        ip  = pkt[IP]
        ts  = float(pkt.time)
        size = len(pkt)

        # Determine flow key and label from port
        if pkt.haslayer(TCP):
            t   = pkt[TCP]
            key = (ip.src, ip.dst, t.sport, t.dport, "tcp")
            port = t.dport
        elif pkt.haslayer(UDP):
            u   = pkt[UDP]
            key = (ip.src, ip.dst, u.sport, u.dport, "udp")
            port = u.dport
        else:
            continue

        label = PORT_TO_LABEL.get(port)
        if label is None:
            continue  # skip non-training traffic

        flows[key]["times"].append(ts)
        flows[key]["sizes"].append(size)
        flows[key]["label"] = label

    # Extract features per flow
    X, y = [], []
    for key, data in flows.items():
        ts  = np.array(data["times"])
        sz  = np.array(data["sizes"])
        lbl = data["label"]

        if len(ts) < 2:     # skip flows with too few packets
            continue

        duration    = ts[-1] - ts[0] + 1e-9
        iats        = np.diff(ts)
        total_bytes = sz.sum()

        byte_rate   = total_bytes / duration
        pkt_rate    = len(ts) / duration
        mean_iat    = np.mean(iats)
        std_iat     = np.std(iats)
        mean_size   = np.mean(sz)
        burst_ratio = np.mean(iats < 0.001)

        X.append([byte_rate, pkt_rate, mean_iat,
                  std_iat, mean_size, burst_ratio])
        y.append(lbl)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"[*] Extracted {len(X)} flow samples")
    for i, name in enumerate(LABEL_NAMES):
        count = int((y == i).sum())
        print(f"    {name:<12}: {count} flows")

    return X, y


if __name__ == "__main__":
    import os
    os.makedirs("weights", exist_ok=True)

    X, y = extract_features_from_pcap("/tmp/training_capture.pcap")

    np.save("train_X.npy", X)
    np.save("train_y.npy", y)
    print("[*] Saved train_X.npy and train_y.npy")
