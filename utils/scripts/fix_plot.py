import nbformat
import sys

with open('NMNIST_SNN_Analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# 1. Update the plot code 
for cell in nb.cells:
    if 'def diagnose_models' in cell.source and 'df_pivot.plot' in cell.source:
        lines = cell.source.split('\n')
        new_lines = []
        for line in lines:
            if 'ax = df_pivot.plot(kind=' in line:
                new_lines.append(line)
                new_lines.append(
                    '    # Add Data Labels\n'
                    '    for container in ax.containers:\n'
                    '        ax.bar_label(container, fmt="%.1f", label_type="edge", padding=3)\n'
                )
            elif 'plt.ylabel(' in line:
                new_lines.append(line)
                new_lines.append('    plt.grid(True, axis="y", linestyle="--", alpha=0.7)')
            else:
                new_lines.append(line)
        cell.source = '\n'.join(new_lines)


# 2. Add the Markdown block at the end if it's not already there
conclusion_source = '''### Diagnostic Conclusion

Based on the evaluated metrics:

- **H1 (Timestep cost):** Supported. `T=16` acts as a multiplier on simulation overhead, making recurrent/state updates expensive.
- **H2 (Sparsity):** Not supported. Spike activity is sparse enough.
- **H3 (Runtime overhead):** Supported. SNN forward latency is substantial (65.86 ms per batch) due to unoptimized surrogate gradient state tracking overhead.
- **H4 (Early Temporal Collapse):** Strongly Supported. The Hybrid model extracts spikes for only 2 layers, collapses the time dimension with `mean(0)`, and processes the rest as Dense ANN operations (costing 0.263M MACs). This completely bypasses simulating `T=16` timesteps for the spatial classifier!

**Final Verdict:** The Hybrid model is drastically cheaper because it exits the temporal domain early (H4), bypassing the strict per-timestep overhead (H1 and H3) for the deep linear classifier layers.'''

last_cell = nb.cells[-1]
if last_cell.cell_type != 'markdown' or '### Diagnostic Conclusion' not in last_cell.source:
    new_md = nbformat.v4.new_markdown_cell(source=conclusion_source)
    nb.cells.append(new_md)


with open('NMNIST_SNN_Analysis.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Formatting applied.")
