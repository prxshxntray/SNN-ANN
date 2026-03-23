import subprocess
import time

def run_host(net, host_name, cmd, background=True):
    h = net.get(host_name)
    if background:
        h.cmd(f"{cmd} &")
    else:
        h.cmd(cmd)

def generate_elephant(net, src, dst, dst_ip, port, duration=30):
    """ High bandwidth TCP - iperf long flow """
    run_host(net, dst, f"iperf -s -p {port}")
    time.sleep(0.5)
    run_host(net, src, f"iperf -c {dst_ip} -p {port} -t {duration} -b 500M")

def generate_mouse(net, src, dst, dst_ip, port, count=20):
    """ Many small short TCP flows """
    run_host(net, dst, f"iperf -s -p {port}")
    time.sleep(0.5)
    for _ in range(count):
        run_host(net, src, f"iperf -c {dst_ip} -p {port} -t 1 -b 1M", background=False)
        time.sleep(0.2)

def generate_gpu_task(net, src, dst, dst_ip, port, duration=15):
    """ Large Bursty UDP - mimics RDMA / GPU-to-GPU """
    run_host(net, dst, f"iperf -s -p {port}")
    time.sleep(0.5)
    run_host(net, src, f"iperf -c {dst_ip} -p {port} -t {duration} -b 900M")

def generate_interactive(net, src, dst, dst_ip, port, count=50):
    """ Small low-latency flows - mimics SSH/HTTP """
    run_host(net, dst, f"iperf -s -p {port}")
    time.sleep(0.5)
    for _ in range(count):
        run_host(net, src, f"iperf -c {dst_ip} -p {port} -t 2 -b 100K", background=False)
        time.sleep(0.2)


print("[*] Starting traffic generation for training data...")

print("[*] Elephant Flows (h1->h2)...")
generate_elephant(net, "h1", "h2", "10.0.0.2", 5101, duration=30)

print("[*] Mouse Flows (h3->h4)...")
generate_mouse(net, "h3", "h4", "10.0.1.2", 5102, count=30)

print("[*] GPU Task Flows (h5->h6)...")
generate_gpu_task(net, "h5", "h6", "10.1.0.2", 5103, duration=15)

print("[*] Interactive Flows (h7->h8)...")
generate_interactive(net, "h7", "h8", "10.1.1.2", 5104, count=50)

print("[*] Done. Check /tmp/training_capture.pcap")


