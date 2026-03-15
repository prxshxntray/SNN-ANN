import json

notebook_path = "/Users/rakeshkrai/Library/Mobile Documents/com~apple~CloudDocs/Wattr/Datasets /SNN-ANN/N-MNIST/NMNIST_SNN_Analysis.ipynb"

# 1. New evaluate_latency_v2 function
evaluate_latency_v2_code = [
    "def evaluate_latency_v2(model, loader, model_type, confidence_threshold=0.8):\n",
    "    model.eval()\n",
    "    correct_at_t = np.zeros(T)\n",
    "    total = 0\n",
    "    \n",
    "    # Metrics\n",
    "    time_to_decision = []\n",
    "    confidence_over_time = [] # List of (T, B) arrays to be stacked later\n",
    "    \n",
    "    with torch.no_grad():\n",
    "        for inputs, labels in loader:\n",
    "            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)\n",
    "            batch_size = labels.size(0)\n",
    "            total += batch_size\n",
    "            \n",
    "            # Initialize outputs container (T, B, 10)\n",
    "            all_step_outputs = torch.zeros(T, batch_size, 10, device=DEVICE)\n",
    "\n",
    "            if model_type == 'snn':\n",
    "                outputs = model(inputs)\n",
    "                # SNN output is (T, B, 10) membrane potential traces\n",
    "                # Logits at time t = mean(outputs[0...t])\n",
    "                # Efficient cumulative mean:\n",
    "                outs_cum = outputs.cumsum(dim=0)\n",
    "                div = torch.arange(1, T+1, device=DEVICE).view(T, 1, 1)\n",
    "                all_step_outputs = outs_cum / div\n",
    "                functional.reset_net(model)\n",
    "\n",
    "            elif model_type == 'crnn':\n",
    "                # CRNN output is (B, T, 10)\n",
    "                outputs = model(inputs)\n",
    "                # Result at time t is just the output at time t\n",
    "                all_step_outputs = outputs.permute(1, 0, 2)\n",
    "\n",
    "            elif model_type == 'cnn':\n",
    "                # Evaluate CNN on accumulated frames 1..t\n",
    "                for t in range(1, T + 1):\n",
    "                    input_t = inputs[:, :t].sum(dim=1)\n",
    "                    out_t = model(input_t)\n",
    "                    all_step_outputs[t-1] = out_t\n",
    "\n",
    "            elif model_type == 'hybrid':\n",
    "                # Evaluate Hybrid on partial sequence 1..t\n",
    "                for t in range(1, T + 1):\n",
    "                    input_t = inputs[:, :t]\n",
    "                    out_t = model(input_t)\n",
    "                    all_step_outputs[t-1] = out_t\n",
    "                    functional.reset_net(model)\n",
    "\n",
    "            # Calculate accuracy at each t\n",
    "            _, preds = all_step_outputs.max(2)\n",
    "            for t in range(T):\n",
    "                correct_at_t[t] += (preds[t] == labels).sum().item()\n",
    "\n",
    "            # Calculate Time-to-Decision & Confidence Curve\n",
    "            # 1. Apply Softmax to Logits at every step\n",
    "            probs = torch.softmax(all_step_outputs, dim=2) # (T, B, 10)\n",
    "            \n",
    "            # 2. Get Max Probability (Confidence) at every step\n",
    "            max_conf, _ = probs.max(dim=2) # (T, B)\n",
    "            \n",
    "            # 3. Store batch confidence for later averaging\n",
    "            confidence_over_time.append(max_conf.cpu().numpy()) \n",
    "\n",
    "            # 4. Find first t where conf > threshold\n",
    "            met_thresh = (max_conf > confidence_threshold).int()\n",
    "            # argmax returns index of first 1. If no 1s, it returns 0 (which is wrong if never met)\n",
    "            # So we need to handle the \"never met\" case.\n",
    "            # If sum over T is 0, then never met.\n",
    "            has_met = met_thresh.sum(dim=0) > 0\n",
    "            first_times = met_thresh.argmax(dim=0)\n",
    "            \n",
    "            for b in range(batch_size):\n",
    "                if has_met[b]:\n",
    "                    t_dec = first_times[b].item() + 1\n",
    "                else:\n",
    "                    t_dec = T # Penalty for never being confident\n",
    "                time_to_decision.append(t_dec)\n",
    "\n",
    "    # Aggregate Confidence Curves: (N_batches, T, B) -> (T, Total_Samples)\n",
    "    # Actually we just want Median Confidence vs T across all samples\n",
    "    # confidence_over_time is list of (T, B)\n",
    "    all_conf = np.concatenate(confidence_over_time, axis=1) # (T, Total_Samples)\n",
    "    median_conf_vs_t = np.median(all_conf, axis=1) # (T,)\n",
    "\n",
    "    return correct_at_t / total * 100, np.median(time_to_decision), median_conf_vs_t\n"
]

# 2. Update Calls to evaluate_latency
# We need to find where evaluate_latency is called and replace it with v2.
# Pattern: acc_snn, lat_snn = evaluate_latency(snn, test_loader, 'snn')
# New: acc_snn, lat_snn, conf_snn = evaluate_latency_v2(snn, test_loader, 'snn')

updated_calls = [
    "# Run Evaluation (Research Grade)\n",
    "print(\"Evaluating SNN...\")\n",
    "acc_snn, lat_snn, conf_snn = evaluate_latency_v2(snn, test_loader, 'snn')\n",
    "print(\"Evaluating CRNN...\")\n",
    "acc_rnn, lat_rnn, conf_rnn = evaluate_latency_v2(rnn, test_loader, 'crnn')\n",
    "print(\"Evaluating CNN...\")\n",
    "acc_cnn, lat_cnn, conf_cnn = evaluate_latency_v2(cnn, test_loader, 'cnn')\n",
    "print(\"Evaluating Hybrid...\")\n",
    "acc_hyb, lat_hyb, conf_hyb = evaluate_latency_v2(hybrid, test_loader, 'hybrid')\n"
]

# 3. New Visualization Cell
viz_code = [
    "# Plot: Confidence over Time\n",
    "plt.figure(figsize=(10, 6))\n",
    "t_steps = np.arange(1, T+1)\n",
    "plt.plot(t_steps, conf_snn, marker='o', label='SNN')\n",
    "plt.plot(t_steps, conf_rnn, marker='s', label='CRNN')\n",
    "plt.plot(t_steps, conf_cnn, linestyle='--', label='CNN')\n",
    "plt.plot(t_steps, conf_hyb, marker='^', label='Hybrid')\n",
    "plt.axhline(y=0.8, color='r', linestyle=':', label='Decision Threshold (0.8)')\n",
    "plt.xlabel(\"Time Step\")\n",
    "plt.ylabel(\"Median Max Probability (Confidence)\")\n",
    "plt.title(\"Confidence Accumulation over Time\")\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()\n",
    "\n",
    "# Updated Summary Table\n",
    "data = {\n",
    "    \"Model\": [\"SNN\", \"RNN\", \"CNN\", \"Hybrid\"],\n",
    "    \"Final Accuracy\": [acc_snn[-1], acc_rnn[-1], acc_cnn[-1], acc_hyb[-1]],\n",
    "    \"Median Time-to-Decide (tau=0.8)\": [lat_snn, lat_rnn, lat_cnn, lat_hyb],\n",
    "    \"Avg Spikes/Sample\": [energy_snn[0], energy_rnn[0], energy_cnn[0], energy_hyb[0]],\n",
    "}\n",
    "df = pd.DataFrame(data)\n",
    "print(df)\n"
]

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# A. Replace evaluate_latency definition
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "def evaluate_latency(" in source:
            print("Replacing evaluate_latency with v2...")
            cell['source'] = evaluate_latency_v2_code
            break

# B. Replace calls
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "acc_snn, lat_snn = evaluate_latency(" in source:
            print("Updating evaluation calls...")
            cell['source'] = updated_calls
            break

# C. Append Visualization Cell (replacing the old plotting cell if possible, or just append)
# We look for the cell that did the old plotting
found_plot = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "plt.title(\"Latency Analysis: Accuracy vs. Computation Time\")" in source:
             # We actually want to KEEP the accuracy plot, but ADD the confidence plot.
             # So let's append the new visualization code to this cell's source.
             print("Appending Confidence Plot to existing Visualization cell...")
             cell['source'].extend(["\n"] + viz_code)
             found_plot = True
             break

if not found_plot:
    print("Could not find plotting cell. Appending new cell.")
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": viz_code
    }
    nb['cells'].append(new_cell)

with open(notebook_path, 'w') as f:
    json.dump(nb, f)
print("Notebook patched with Research-Grade improvements.")
