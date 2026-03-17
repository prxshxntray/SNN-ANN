import json

try:
    with open('analysis/pjm_hybrid/pjm_hybrid.ipynb', 'r') as f:
        nb = json.load(f)

    errors = False
    for i, cell in enumerate(nb['cells']):
        if 'outputs' in cell:
            for out in cell['outputs']:
                if out.get('output_type') == 'error':
                    errors = True
                    print(f"Error in cell {i}: {out.get('ename')} - {out.get('evalue')}")
                    print("".join(out.get('traceback', []))[:1000])

    if not errors:
        print("No Python exceptions found in execution.")
except Exception as e:
    print(f"Failed to read notebook: {e}")
