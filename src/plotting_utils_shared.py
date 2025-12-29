
from utils_shared import get_dir
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import matplotlib

def latexify_plot(fontsize) -> None:
    text_usetex = True if shutil.which('latex') else False
    params = {
            'text.latex.preamble': r"\usepackage{gensymb} \usepackage{amsmath}",
            'axes.labelsize': fontsize,
            'axes.titlesize': fontsize,
            'legend.fontsize': fontsize,
            'xtick.labelsize': fontsize,
            'ytick.labelsize': fontsize,
            'text.usetex': text_usetex,
            'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)
    return

def barplot(
    approach_labels,
    mean_costs,
    mean_solve_times,
    mean_solve_time_per_iter,
    subpath="diff_drive_results.pgf",
    latexify=False,
    save=False,
    figsize=(8, 10),
    fontsize=12,
):
    plots_dir = get_dir("plots")
    if latexify:
        latexify_plot(fontsize=fontsize)
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, constrained_layout=True)

    axes[0].bar(approach_labels, mean_costs)
    axes[0].axhline(y=mean_costs[0], color='red', linestyle='--', linewidth=2, label='Baseline (Approach 0)')
    axes[0].set_ylabel("Mean Stage Cost")
    axes[0].set_title("Mean Stage Cost")
    axes[0].set_yscale('log')
    axes[0].grid(True, axis='y')
    axes[0].legend(loc='upper right')
    axes[0].text(-0.08, 1.03, '(a)', transform=axes[0].transAxes, fontsize=fontsize, fontweight='bold')

    axes[1].bar(approach_labels, mean_solve_times)
    axes[1].set_ylabel("Mean Solve Time [s]")
    axes[1].set_title("Mean Solve Time")
    axes[1].set_yscale('log')
    axes[1].grid(True, axis='y')
    axes[1].text(-0.08, 1.03, '(b)', transform=axes[1].transAxes, fontsize=fontsize, fontweight='bold')

    axes[2].bar(approach_labels, mean_solve_time_per_iter)
    axes[2].set_ylabel("Mean Solve Time per Iteration [s]")
    axes[2].set_title("Mean Solve Time per SQP Iteration")
    axes[2].set_yscale('log')
    axes[2].set_xticks(range(len(approach_labels)))
    axes[2].set_xticklabels(approach_labels, rotation=30, ha='right')
    axes[2].tick_params(axis='x', labelsize=10, pad=2)
    axes[2].grid(True, axis='y')
    axes[2].text(-0.08, 1.03, '(c)', transform=axes[2].transAxes, fontsize=fontsize, fontweight='bold')

    fig.align_ylabels(axes)

    if save:
        out_path = plots_dir / subpath
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")

    plt.show()

def pareto_frontier(
    mean_solve_times,
    mean_costs,
    mean_costs_baseline,
    approach_labels,
    markers,
    marker_sizes,
    subpath="pareto_front_symbols_colors_betterleg_highlight1.pdf",
    latexify=True,
    save=False,
    figsize=(6, 6),
    fontsize=12,
    x_lim = None,
    x_nonlog=False,
    y_nonlog=False,
    x_lim_upper = None,
    legend=True
):
    cmap = plt.get_cmap('tab10')
    colors = cmap.colors[:7]

    if latexify:
        latexify_plot(fontsize=fontsize)

    with plt.rc_context({'text.usetex': False}):
        fig, ax = plt.subplots(figsize=figsize)

        mean_costs_perc = 100*(mean_costs - mean_costs_baseline)/mean_costs_baseline
        for idx, (xi, yi, lbl) in enumerate(zip(mean_solve_times, mean_costs_perc, approach_labels)):
            if len(approach_labels) < 7:
                if idx == 1:
                    idx_color = 6
                elif idx == 5:
                    idx_color = 1
                elif idx >= 3:
                    idx_color = idx + 1
                else:
                    idx_color = idx

                if idx >= 3:
                    idx_marker = idx + 1
                else:
                    idx_marker = idx
            else: 
                # color index logic
                if idx == 1:
                    idx_color = 6
                elif idx == 6:
                    idx_color = 1
                else:
                    idx_color = idx

                idx_marker = idx

            color = colors[idx_color]
            marker = markers[idx_marker]
            size = marker_sizes[idx_marker]

            ax.scatter(xi, yi,
                       color=color,
                       marker=marker,
                       s=size,
                       label=lbl)
        if not x_nonlog:
            ax.set_xscale('log')
        if not y_nonlog:
            ax.set_yscale('log')    
        ax.set_xlabel('Mean solve time [s]')
        ax.set_ylabel(r"Mean cost increase [%]")
        #ax.set_title('Pareto Frontier')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        if legend:
            ax.legend(loc='best', frameon=True)
        if x_lim is not None:
            ax.set_xlim(left=x_lim)
        if x_lim_upper is not None:
            ax.set_xlim(right=x_lim_upper)

        if save:
            plots_dir = Path(get_dir("plots"))
            out_path = plots_dir / subpath
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, bbox_inches='tight')

        plt.show()