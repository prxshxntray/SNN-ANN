import json

notebook_path = "/Users/rakeshkrai/Library/Mobile Documents/com~apple~CloudDocs/Wattr/Datasets /SNN-ANN/N-MNIST/NMNIST_SNN_Analysis.ipynb"

# Robust measure_energy function
measure_energy_source = [
    "def measure_energy(model, loader, model_type):\n",
    "    global total_spikes\n",
    "    total_spikes = 0\n",
    "    hooks = []\n",
    "    \n",
    "    # Register hooks for SNN/Hybrid\n",
    "    if model_type in ['snn', 'hybrid']:\n",
    "        for name, module in model.named_modules():\n",
    "            if isinstance(module, neuron.LIFNode):\n",
    "                hooks.append(module.register_forward_hook(spike_hook))\n",
    "    \n",
    "    # Ensure model is on the correct device BEFORE running\n",
    "    model = model.to(DEVICE)\n",
    "    model.eval()\n",
    "    num_samples = 0\n",
    "    \n",
    "    with torch.no_grad():\n",
    "        for inputs, _ in loader:\n",
    "            inputs = inputs.to(DEVICE)\n",
    "            model(inputs)\n",
    "            if model_type in ['snn', 'hybrid']: \n",
    "                functional.reset_net(model)\n",
    "            num_samples += inputs.size(0)\n",
    "            break # Just one batch for simple estimation\n",
    "            \n",
    "    # Remove hooks\n",
    "    for h in hooks:\n",
    "        h.remove()\n",
    "        \n",
    "    avg_spikes = total_spikes / num_samples\n",
    "    \n",
    "    # Estimated MACs (using thop)\n",
    "    try:\n",
    "        # Use dummy input on correct device\n",
    "        inputs = torch.randn(1, T, 2, 34, 34).to(DEVICE)\n",
    "        if model_type == 'snn':\n",
    "            # For SNN, thop gives MACs assuming dense. We scale by sparsity.\n",
    "            # But real energy is ~ spikes * fan-out. \n",
    "            # We will just report Spikes.\n",
    "            ops = 0\n",
    "        else:\n",
    "             # RNN/CNN MACs\n",
    "            # Thop might reset device if not careful? Usually okay.\n",
    "            # Just run it.\n",
    "            ops, _ = thop.profile(model, inputs=(inputs,), verbose=False)\n",
    "    except Exception as e:\n",
    "        print(f\"Thop profiling failed for {model_type}: {e}\")\n",
    "        ops = 0\n",
    "        \n",
    "    return avg_spikes, ops\n"
]

with open(notebook_path, 'r') as f:
    nb = json.load(f)

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if "def measure_energy(" in source_str:
            print("Found measure_energy definition. Replacing with robust version.")
            cell['source'] = measure_energy_source
            found = True
            break

if found:
    with open(notebook_path, 'w') as f:
        json.dump(nb, f)
    print("Notebook updated: measure_energy fixed.")
else:
    print("Could not find measure_energy definition.")
