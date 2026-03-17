
import numpy as np
import os

# Path from previous turns
DATASET_PATH = "/Users/rakeshkrai/Library/Mobile Documents/com~apple~CloudDocs/Wattr/Datasets /SNN-ANN/N-MNIST"
SAMPLE_FILE = os.path.join(DATASET_PATH, "Train", "0", "00002.bin")

def read_nmnist_file(filename):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return None

    with open(filename, "rb") as f:
        # Read all bytes
        raw_data = np.fromfile(f, dtype=np.uint8)
    
    if len(raw_data) == 0:
        print("Empty file")
        return None

    # Check if length is divisible by 5
    if len(raw_data) % 5 != 0:
        print(f"Warning: File length {len(raw_data)} is not divisible by 5. Truncating.")
        raw_data = raw_data[:-(len(raw_data) % 5)]
        
    num_events = len(raw_data) // 5
    data = raw_data.reshape(num_events, 5)

    # Convert to standard types
    # x: byte 0
    # y: byte 1
    # p: byte 2 (1 bit)
    # ts: byte 2 (7 bits) + byte 3 + byte 4
    
    # MATLAB:
    # TD.x    = evtStream(1:5:end); 
    # TD.y    = evtStream(2:5:end);
    # TD.p    = bitshift(evtStream(3:5:end), -7);
    # TD.ts   = bitshift(bitand(evtStream(3:5:end), 127), 16); 
    # TD.ts   = TD.ts + bitshift(evtStream(4:5:end), 8);
    # TD.ts   = TD.ts + evtStream(5:5:end);
    
    x = data[:, 0]
    y = data[:, 1]
    p = (data[:, 2] >> 7)
    
    ts = ((data[:, 2] & 0x7F).astype(np.int32) << 16) | \
         (data[:, 3].astype(np.int32) << 8) | \
         (data[:, 4].astype(np.int32))
         
    return x, y, p, ts

if __name__ == "__main__":
    print(f"Checking file: {SAMPLE_FILE}")
    try:
        result = read_nmnist_file(SAMPLE_FILE)
        if result:
            x, y, p, ts = result
            print(f"Success! Read {len(ts)} events.")
            print(f"X: min={x.min()}, max={x.max()}")
            print(f"Y: min={y.min()}, max={y.max()}")
            print(f"P: min={p.min()}, max={p.max()}")
            print(f"TS: min={ts.min()}, max={ts.max()}")
            print("First 5 events:")
            for i in range(5):
                print(f"  {i}: x={x[i]}, y={y[i]}, p={p[i]}, ts={ts[i]}")
    except Exception as e:
        print(f"Error: {e}")
