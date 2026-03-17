import json

notebook_path = "/Users/rakeshkrai/Library/Mobile Documents/com~apple~CloudDocs/Wattr/Datasets /SNN-ANN/N-MNIST/NMNIST_SNN_Analysis.ipynb"

# The new logic for load_or_train to handle various model names correctly
# We prepend this to the cell that calls it.
new_load_or_train_source = [
    "def load_or_train(model, model_name, train_loader, epochs=5):\n",
    "    path = os.path.join(MODELS_DIR, f\"{model_name}.pth\")\n",
    "    \n",
    "    if os.path.exists(path):\n",
    "        print(f\"Loading {model_name} from {path}...\")\n",
    "        state_dict = torch.load(path, map_location='cpu')\n",
    "        model.load_state_dict(state_dict, strict=False)\n",
    "        model = model.to(DEVICE)\n",
    "        return model\n",
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
    "            # Loss computation handles different output improvements\n",
    "            # SNN: (T, B, 10) -> Mean over T\n",
    "            # RNN/CRNN: (B, T, 10) -> Last Step\n",
    "            # Others (CNN, Hybrid, old RNN?): (B, 10)\n",
    "            \n",
    "            if 'snn' in model_name and 'hybrid' not in model_name:\n",
    "                loss = criterion(outputs.mean(0), labels)\n",
    "            elif 'rnn' in model_name:\n",
    "                loss = criterion(outputs[:, -1, :], labels)\n",
    "            else:\n",
    "                loss = criterion(outputs, labels)\n",
    "                \n",
    "            loss.backward()\n",
    "            optimizer.step()\n",
    "            \n",
    "            total_loss += loss.item()\n",
    "            if 'snn' in model_name and 'hybrid' not in model_name:\n",
    "                functional.reset_net(model)\n",
    "                \n",
    "        print(f\"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}\")\n",
    "        \n",
    "    torch.save(model.state_dict(), path)\n",
    "    return model\n",
    "\n"
]

with open(notebook_path, 'r') as f:
    nb = json.load(f)

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        # Identify the cell that calls load_or_train but hasn't defined it yet
        # We look for the call signature being present
        if 'snn = load_or_train(snn, "snn_model_tuned"' in source_str:
            print("Found cell calling load_or_train.")
            
            # Check if definition is already present to avoid duplication
            if "def load_or_train(model, model_name" in source_str:
                print("load_or_train is already defined in this cell. Checking replace vs append.")
                # We will just replace the whole definition if needed, or assume manual fix needed.
                # For now, let's assume if it is there, we might need to update it.
                # But matching logic is harder. Let's assume if it's NOT there we add it.
                # If it IS there, we failed previously to find it via grep.
            else:
                print("load_or_train definition MISSING in this cell. Prepending it.")
                cell['source'] = new_load_or_train_source + cell['source']
                found = True
                break

if found:
    with open(notebook_path, 'w') as f:
        json.dump(nb, f)
    print("Notebook updated: load_or_train definition restored.")
else:
    print("Could not find suitable cell to patch.")
