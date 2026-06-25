import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob


def parse_file(filename):
    bfs = []
    sssp = []
    with open(filename, 'r') as f:
        for line in f:
            m = re.match(r'Time for BFS \d+ is ([\d\.]+)', line)
            if m:
                bfs.append(float(m.group(1)))
            m = re.match(r'Time for SSSP \d+ is ([\d\.]+)', line)
            if m:
                sssp.append(float(m.group(1)))
    return np.array(bfs), np.array(sssp)


def main():
    results_dir = os.path.join(os.path.dirname(__file__), '../experiments/results')
    dirs = sorted(glob.glob(os.path.join(results_dir, 'graph500_s*')))

    # Collect raw times per (node_count, algo, mode)
    records = {}   # (node_count, algo) -> {mode: [times]}

    for d in dirs:
        dirname = os.path.basename(d)
        m = re.match(r'graph500_s(\d+)_np(\d+)_.*', dirname)
        if not m:
            continue
        node_count = int(m.group(2)) // 64
        if node_count not in [1, 2, 4]:
            continue

        try:
            bfs_v, sssp_v = parse_file(os.path.join(d, '00_vanilla_run.out'))
            bfs_p, sssp_p = parse_file(os.path.join(d, '01_profile_run.out'))
            bfs_b, sssp_b = parse_file(os.path.join(d, '02_boosted_run.out'))
        except FileNotFoundError:
            continue

        if len(bfs_v) == 0 or len(bfs_p) == 0 or len(bfs_b) == 0:
            continue

        for algo, v, p, b in [('BFS', bfs_v, bfs_p, bfs_b),
                               ('SSSP', sssp_v, sssp_p, sssp_b)]:
            key = (node_count, algo)
            records[key] = {
                'Baseline':  v,
                'Profile':   p,
                'Reordered': b,
            }

    # Compute speedup = mean(Baseline) / mean(other)  per run
    # For per-sample speedup we pair element-wise if lengths match,
    # otherwise fall back to ratio of means.
    # Baseline is fixed at 1.0 but its error bar shows the coefficient of
    # variation (std/mean) of the raw times — the same relative uncertainty
    # that propagates into the other speedup ratios.
    data = []
    for (node_count, algo), modes in records.items():
        base = modes['Baseline']
        nodes_label = f'{node_count} Node{"s" if node_count > 1 else ""}'
        if len(base) > 0:
            cv = base.std() / base.mean()   # relative spread around 1.0
            data.append({
                'Nodes':     nodes_label,
                'Algorithm': algo,
                'Mode':      'Baseline',
                'Speedup':   1.0,
                'Speedup_std': cv,
            })
        for mode in ['Profile', 'Reordered']:
            other = modes[mode]
            if len(base) == len(other) and len(base) > 0:
                speedups = base / other          # element-wise
            elif len(base) > 0 and len(other) > 0:
                speedups = np.array([base.mean() / other.mean()])
            else:
                continue
            data.append({
                'Nodes':     nodes_label,
                'Algorithm': algo,
                'Mode':      mode,
                'Speedup':   speedups.mean(),
                'Speedup_std': speedups.std(),
            })

    df = pd.DataFrame(data)

    # ── Palette ──────────────────────────────────────────────────────────────
    PASTEL = {
        'Baseline':  '#D4918C',   # dusty rose
        'Profile':   '#8FB098',   # soft sage
        'Reordered': '#8FADD1',   # muted periwinkle
    }
    BG    = '#fafafa'
    GRID  = '#ebebeb'
    TEXT  = '#3d3d3d'
    TITLE = '#2a2a2a'

    # ── rcParams ─────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':       'sans-serif',
        'font.sans-serif':   ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.spines.left':  False,
        'axes.spines.bottom':False,
        'axes.grid':         True,
        'axes.grid.axis':    'y',
        'grid.color':        GRID,
        'grid.linewidth':    0.8,
        'axes.facecolor':    BG,
        'figure.facecolor':  BG,
        'text.color':        TEXT,
        'axes.labelcolor':   TEXT,
        'xtick.color':       TEXT,
        'ytick.color':       TEXT,
        'xtick.bottom':      False,
    })

    node_order = ['1 Node', '2 Nodes', '4 Nodes']
    mode_order = ['Baseline', 'Profile', 'Reordered']
    algo_order = ['BFS', 'SSSP']

    fig, axes = plt.subplots(
        nrows=2, ncols=3,
        figsize=(13, 7),
        sharey=False,
        facecolor=BG,
    )
    fig.subplots_adjust(top=0.84, hspace=0.52, wspace=0.32,
                        left=0.09, right=0.97)

    for row_idx, algo in enumerate(algo_order):
        df_algo = df[df['Algorithm'] == algo]

        for col_idx, nodes in enumerate(node_order):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor(BG)

            df_sub = df_algo[df_algo['Nodes'] == nodes]

            x_positions = np.arange(len(mode_order))
            bar_width   = 0.52

            for xi, mode in enumerate(mode_order):
                row = df_sub[df_sub['Mode'] == mode]
                if len(row) == 0:
                    continue
                mean = row['Speedup'].values[0]
                std  = row['Speedup_std'].values[0]

                ax.bar(
                    xi, mean,
                    width=bar_width,
                    color=PASTEL[mode],
                    linewidth=0,
                    zorder=3,
                )
                ax.bar(
                    xi, mean,
                    width=bar_width,
                    color='none',
                    edgecolor='white',
                    linewidth=1.2,
                    zorder=4,
                )
                if std > 0:
                    ax.errorbar(
                        xi, mean, yerr=std,
                        fmt='none',
                        ecolor='#888888',
                        elinewidth=1.0,
                        capsize=4,
                        capthick=1.0,
                        zorder=5,
                    )

            # Reference line at speedup = 1 (no improvement)
            ax.axhline(1.0, color='#bbbbbb', linewidth=0.9,
                       linestyle='--', zorder=2)

            ax.set_xticks(x_positions)
            ax.set_xticklabels(mode_order, fontsize=9.5)
            ax.tick_params(axis='y', labelsize=8.5)
            ax.set_xlabel('')

            ax.set_title(nodes, fontsize=11, fontweight='500',
                         color=TITLE, pad=10)

            ax.set_ylabel('Speedup', fontsize=9.5, labelpad=8, color=TEXT)

    # ── Row titles: horizontal, above the leftmost plot in each row ─────────
    for row_idx, algo in enumerate(algo_order):
        ax_left = axes[row_idx][0]
        ax_left.annotate(
            algo,
            xy=(0, 1), xycoords='axes fraction',
            xytext=(0, 28), textcoords='offset points',
            ha='left', va='bottom',
            fontsize=13, fontweight='normal',
            color=TITLE,
        )

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=PASTEL[m], label=m, linewidth=0)
        for m in mode_order
    ]
    fig.legend(
        handles=legend_handles,
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 0.97),
        frameon=False,
        fontsize=10,
        handlelength=1.4,
        handleheight=0.9,
        borderaxespad=0,
    )

    fig.text(
        0.5, 1.01,
        'Graph500',
        ha='center', va='bottom',
        fontsize=14, fontweight='normal',
        color=TITLE,
        transform=fig.transFigure,
    )

    images_dir = os.path.join(os.path.dirname(__file__), 'images')
    os.makedirs(images_dir, exist_ok=True)
    out_path = os.path.join(images_dir, 'speedup_plot.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=BG)
    print(f'Plot saved to {out_path}')


if __name__ == '__main__':
    main()
