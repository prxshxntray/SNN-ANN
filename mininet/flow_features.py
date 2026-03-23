import time
import numpy as np
from collections import defaultdict, deque

class FlowTracker: 
    """
    Tracks per-flow statistics. A flow is identified by the 5-tuple:
    (src_ip, dst_ip, src_port, dst_port, protocol)
    """

    def __init__(self, window_seconds=1.0, max_history=100):
        self.window = window_seconds
        self.flows = defaultdict(lambda: {
            "timestamps": deque(maxlen=max_history),
            "sizes": deque(maxlen=max_history),
            "total_bytes": 0, 
            "first_seen": None,
            "last_seen": None,
        })

    def record_packet(self, flow_key, pkt_size, timestamp=None):
        ts = timestamp or time.time()
        f = self.flows[flow_key]
        f["timestamps"].append(ts)
        f["sizes"].append(pkt_size)
        f["total_bytes"] += pkt_size
        f["last_seen"] = ts
        if f["first_seen"] is None:
            f["first_seen"] = ts

    def extract_features(self, flow_key) -> np.ndarray:
        """
        Returns a 6-dimensional feature vector:
        [byte_rate, pkt_rate, mean_iat, std_iat, mean_pkt_size, burst_ratio]
        """

        f = self.flows[flow_key]
        ts = np.array(f["timestamps"])
        sz = np.array(f["sizes"])

        if len(ts) < 2: 
            return np.zeros(6)
        
        duration = ts[-1] - ts[0] + 1e-9
        iats = np.diff(ts) # inter-arrival times

        byte_rate = f["total_bytes"] / duration
        pkt_rate = len(ts) / duration
        mean_iat = np.mean(iats)
        std_iat = np.std(iats)
        mean_size = np.mean(sz)

        burst_ratio = np.mean(iats < 0.001)

        return np.array([
            byte_rate, 
            pkt_rate,
            mean_iat, 
            std_iat,
            mean_size,
            burst_ratio,
            ], dtype=np.float32)
