
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from snr.dataloader import get_slice

def plot_task_curves(ax: plt.Axes, task, signal_label, plotted_sizes, plotted_mixes, metric, df, colors, SEED, task_idx,
                     n_mixes_label="25 corpora", xc_label="5xC", signal_size=None):
    """Plot training curves for a single task on the given axis.

    n_mixes_label: legend description of the curve family (default "25 corpora" for
        DataDecide; pass e.g. "3 data mixtures" for Apertus).
    xc_label: trailing text on the per-size annotation (default "5xC" for DataDecide
        Chinchilla-multiple labels; pass "" to omit when not applicable).
    signal_size: which size's curves to bracket as the "signal" range. Defaults to
        the largest size in ``plotted_sizes`` (or ``"1B"`` if present, for backward compat).
    """
    if signal_size is None:
        signal_size = "1B" if "1B" in plotted_sizes else plotted_sizes[-1]
    
    # Track best final performance for each size
    size_best_perf = {size: {'value': float('-inf'), 'x': None, 'y': None} for size in plotted_sizes}
    final_values_signal = []
    lines = []

    for mix_idx, mix in enumerate(plotted_mixes):
        for size in plotted_sizes:
            curve_data = get_slice(df, mix=mix, task=task, size=size, seed=SEED)

            # Remove first X% of rows
            curve_data = curve_data[curve_data['compute'] > 0].sort_values('compute') # remove compute=0
            curve_data = curve_data.iloc[int(0.05*len(curve_data)):]

            line = ax.plot(
                curve_data['compute'], curve_data[metric], linewidth=0.5,
                color=colors[mix_idx], alpha=0.5
            )
            ax.scatter(
                curve_data['compute'].iloc[-1], curve_data[metric].iloc[-1],
                color=colors[mix_idx], alpha=0.5, s=10, marker='x'
            )

            # Keep final values for rank correlation
            if size == signal_size:
                final_values_signal.append((mix, curve_data[metric].iloc[-1]))
            
            # Keep track of best performing mix
            final_value = curve_data[metric].iloc[-1]
            if final_value > size_best_perf[size]['value']:
                size_best_perf[size]['value'] = final_value
                size_best_perf[size]['x'] = curve_data['compute'].iloc[-1]
                size_best_perf[size]['y'] = final_value
        
        if task_idx == 0:
            lines.extend(line)

    # Add size annotations
    suffix = f'-{xc_label}' if xc_label else ''
    for size, perf in size_best_perf.items():
        if perf['x'] is not None:
            y_offset = perf['y'] * 0.05 if size in ['20M', '60M'] else 0
            ax.annotate(f'{size}{suffix}  ', (perf['x'], perf['y'] + y_offset),
                        xytext=(0, 0), textcoords='offset points', fontsize=8, ha='right', va='top')

    ############
    # Add bracket
    ############
    if final_values_signal:
        max_compute = curve_data['compute'].iloc[-1]
        values = [score for (mix, score) in final_values_signal]
        y_min = min(values)
        y_max = max(values)
            
        y_mid = (y_min + y_max) / 2

        x_pos = max_compute * 1.15
        bracket_width = x_pos * 0.05
        text_spacing = bracket_width

        ax.plot([x_pos, x_pos], [y_min, y_max], color='black', linewidth=1)
        ax.plot([x_pos, x_pos - bracket_width], [y_min, y_min], color='black', linewidth=1)
        ax.plot([x_pos, x_pos - bracket_width], [y_max, y_max], color='black', linewidth=1)
        
        ax.annotate(signal_label,
                    xy=(x_pos + text_spacing, y_mid),
                    xytext=(x_pos + text_spacing * 2, y_mid),
                    ha='left', va='center')
    ############

    ax.set_xscale('log')
    ax.set_xlabel('Compute (FLOPs)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1]*3)

    ax.plot([], [], color='grey', linewidth=0.5, label=f'Training curve\n({n_mixes_label})')
    ax.scatter([], [], color='grey', s=10, marker='x', label='Final checkpoint')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.2)

    return lines