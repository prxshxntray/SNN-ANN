import json

notebook_path = "/Users/rakeshkrai/Library/Mobile Documents/com~apple~CloudDocs/Wattr/Datasets /SNN-ANN/N-MNIST/NMNIST_SNN_Analysis.ipynb"

# 1. Robust load_or_train function
load_or_train_new = [
    "def load_or_train(model, model_name, train_loader, epochs=5):\n",
    "    path = os.path.join(MODELS_DIR, f\"{model_name}.pth\")\n",
    "    \n",
    "    # Try loading if exists\n",
    "    if os.path.exists(path) and model_name != \"cnn_baseline\":\n",
    "        print(f\"Loading {model_name} from {path}...\")\n",
    "        try:\n",
    "            # Load to CPU first to avoid MPS float64/device issues\n",
    "            state_dict = torch.load(path, map_location='cpu')\n",
    "            # strict=False ignores extra keys (like thop's total_ops)\n",
    "            model.load_state_dict(state_dict, strict=False)\n",
    "            model = model.to(DEVICE)\n",
    "            return model\n",
    "        except RuntimeError as e:\n",
    "            print(f\"Failed to load {model_name}: {e}. Retraining...\")\n",
    "    \n",
    "    print(f\"Training {model_name}...\")\n",
    "    model = model.to(DEVICE)\n",
    "    optimizer = torch.optim.Adam(model.parameters(), lr=LR)\n",
    "    criterion = nn.CrossEntropyLoss()\n",
    "    \n",
    "    for epoch in range(epochs):\n",
    "        model.train()\n",
    "        total_loss = 0\n",
    "        correct = 0\n",
    "        total = 0\n",
    "        \n",
    "        for inputs, labels in train_loader:\n",
    "            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)\n",
    "            optimizer.zero_grad()\n",
    "            outputs = model(inputs)\n",
    "            \n",
    "            # Loss computation logic\n",
    "            if 'snn' in model_name and 'hybrid' not in model_name:\n",
    "                # SNN output is spikes/potentials over time. Use mean.\n",
    "                loss = criterion(outputs.mean(0), labels)\n",
    "            elif 'rnn' in model_name:\n",
    "                # RNN output is sequence. Use last step.\n",
    "                loss = criterion(outputs[:, -1, :], labels)\n",
    "            else:\n",
    "                # Hybrid/CNN: standard (B, C)\n",
    "                loss = criterion(outputs, labels)\n",
    "                \n",
    "            loss.backward()\n",
    "            optimizer.step()\n",
    "            \n",
    "            total_loss += loss.item()\n",
    "            if 'snn' in model_name and 'hybrid' not in model_name:\n",
    "                functional.reset_net(model)\n",
    "            if model_name == 'hybrid_model':\n",
    "                functional.reset_net(model)\n",
    "                \n",
    "        print(f\"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}\")\n",
    "        \n",
    "    torch.save(model.state_dict(), path)\n",
    "    return model\n"
]

# 2. Research-Grade Evaluation Function
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
    "            # Initialize logits container (T, B, 10)\n",
    "            all_logits = torch.zeros(T, batch_size, 10, device=DEVICE)\n",
    "\n",
    "            if model_type == 'snn':\n",
    "                outputs = model(inputs)\n",
    "                # SNN Logic: Logits = Accumulating Spike Counts over time\n",
    "                # outputs is (T, B, 10) spikes (0 or 1)\n",
    "                all_logits = outputs.cumsum(dim=0)\n",
    "                functional.reset_net(model)\n",
    "\n",
    "            elif model_type == 'rnn':\n",
    "                # RNN Logic: Output is sequence of logits (B, T, 10)\n",
    "                outputs = model(inputs)\n",
    "                all_logits = outputs.permute(1, 0, 2)\n",
    "\n",
    "            elif model_type == 'cnn':\n",
    "                # CNN Logic: Re-evaluate on accumulated frames\n",
    "                for t in range(1, T + 1):\n",
    "                    input_t = inputs[:, :t].sum(dim=1)\n",
    "                    out_t = model(input_t)\n",
    "                    all_logits[t-1] = out_t\n",
    "\n",
    "            elif model_type == 'hybrid':\n",
    "                # Hybrid Logic: Re-evaluate on partial sequence\n",
    "                for t in range(1, T + 1):\n",
    "                    input_t = inputs[:, :t]\n",
    "                    out_t = model(input_t)\n",
    "                    all_logits[t-1] = out_t\n",
    "                    functional.reset_net(model)\n",
    "\n",
    "            # Calculate accuracy at each t (using current logits)\n",
    "            _, preds = all_logits.max(2)\n",
    "            for t in range(T):\n",
    "                correct_at_t[t] += (preds[t] == labels).sum().item()\n",
    "\n",
    "            # Calculate Time-to-Decision & Confidence Curve\n",
    "            # 1. Apply Softmax to Logits at every step\n",
    "            # Note for SNN: Softmax(SpikeCount). High count = high conf.\n",
    "            probs = torch.softmax(all_logits, dim=2) # (T, B, 10)\n",
    "            \n",
    "            # 2. Get Max Probability (Confidence) at every step\n",
    "            max_conf, _ = probs.max(dim=2) # (T, B)\n",
    "            \n",
    "            # 3. Store batch confidence for later averaging\n",
    "            confidence_over_time.append(max_conf.cpu().numpy()) \n",
    "\n",
    "            # 4. Find first t where conf > threshold\n",
    "            met_thresh = (max_conf > confidence_threshold).int()\n",
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
    "    # Median over samples\n",
    "    all_conf = np.concatenate(confidence_over_time, axis=1) # (T, Total_Samples)\n",
    "    median_conf_vs_t = np.median(all_conf, axis=1) # (T,)\n",
    "\n",
    "    return correct_at_t / total * 100, np.median(time_to_decision), median_conf_vs_t\n"
]

# 3. Updated Visualization Code
viz_code = [
    "# Run Evaluation (Research Grade)\n",
    "print(\"Evaluating (Research Grade)...\")\n",
    "acc_snn, lat_snn, conf_snn = evaluate_latency_v2(snn, test_loader, 'snn')\n",
    "acc_rnn, lat_rnn, conf_rnn = evaluate_latency_v2(rnn, test_loader, 'rnn')\n",
    "acc_cnn, lat_cnn, conf_cnn = evaluate_latency_v2(cnn, test_loader, 'cnn')\n",
    "acc_hyb, lat_hyb, conf_hyb = evaluate_latency_v2(hybrid, test_loader, 'hybrid')\n",
    "\n",
    "# 1. Confidence over Time Plot\n",
    "plt.figure(figsize=(10, 6))\n",
    "t_steps = np.arange(1, T+1)\n",
    "plt.plot(t_steps, conf_snn, marker='o', label='SNN')\n",
    "plt.plot(t_steps, conf_rnn, marker='s', label='RNN')\n",
    "plt.plot(t_steps, conf_cnn, linestyle='--', label='CNN')\n",
    "plt.plot(t_steps, conf_hyb, marker='^', label='Hybrid')\n",
    "plt.axhline(y=0.8, color='r', linestyle=':', label='Threshold (0.8)')\n",
    "plt.xlabel(\"Time Step\")\n",
    "plt.ylabel(\"Median Confidence (Max Prob)\")\n",
    "plt.title(\"Confidence Accumulation over Time\")\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()\n",
    "\n",
    "# 2. Summary Table\n",
    "data = {\n",
    "    \"Model\": [\"SNN\", \"RNN\", \"CNN\", \"Hybrid\"],\n",
    "    \"Final Accuracy\": [acc_snn[-1], acc_rnn[-1], acc_cnn[-1], acc_hyb[-1]],\n",
    "    \"Median Time-to-Decide (0.8)\": [lat_snn, lat_rnn, lat_cnn, lat_hyb],\n",
    "    \"Avg Spikes/Sample\": [energy_snn[0], energy_rnn[0], energy_cnn[0], energy_hyb[0]],\n",
    "}\n",
    "print(pd.DataFrame(data))\n"
]

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# A. Replace load_or_train
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "def load_or_train(" in source:
            print("Replacing load_or_train...")
            cell['source'] = load_or_train_new
            break

# B. Replace evaluate_latency
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "def evaluate_latency(" in source:
            print("Replacing evaluate_latency with v2...")
            cell['source'] = evaluate_latency_v2_code
            break

# C. Update Final Results Cell
# Find the cell that calls evaluate_latency and plots things
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "acc_snn, lat_snn = evaluate_latency(" in source:
            print("Updating result evaluation cell...")
            cell['source'] = viz_code
            break

with open(notebook_path, 'w') as f:
    json.dump(nb, f)
print("Notebook patched successfully.")
