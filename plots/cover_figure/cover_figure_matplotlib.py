"""
This file creates the cover figure for the fading-fidelity MPC approach for slow-fast systems.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from plotting_utils_shared import latexify_plot

# Base plot typography
BASE_FS = 15           # from latexify_plot; controls general text
AXIS_LABEL_FS = 20     # axis labels used in the top schematic; we'll match this for bottom too
latexify_plot(fontsize=BASE_FS)

# === Layout controls ===
ADD_TUNING_PLOT = True        # add the lower (tuning) schematic
PANEL_HSPACE = 0.00           # vertical gap between panels (smaller = closer)
BOTTOM_HEIGHT_RATIO = 0.65    # relative height of bottom panel vs top (smaller = more squashed)

# === Legend position controls (bottom panel) ===
LEGEND_LOC = 'upper right'
LEGEND_BBOX = (0.85, 0.82)  # set to None to disable bbox_to_anchor

# === Global axis origin for the drawn arrows (keep in sync across panels) ===
AXIS_X0 = -0.3

# === Helper: compute geometric schedule ===
def compute_geometric_schedule(dt0, n_steps, total_length):
    def geometric_sum_error(r):
        if r == 1.0:
            return dt0 * n_steps - total_length
        return dt0 * (1 - r**n_steps) / (1 - r) - total_length

    result = root_scalar(geometric_sum_error, bracket=[1.00001, 100], method='brentq')
    if not result.converged:
        raise RuntimeError("Could not find suitable growth rate.")
    r = result.root
    steps = dt0 * r**np.arange(n_steps)
    t_riemann = np.cumsum(np.concatenate(([0.0], steps)))
    return t_riemann

# === Signal setup ===
t_arrow_length = 7.0
x_dot_arrow_length = 3.0
dashed_line_height = 3.0  # <-- adjustable height for all dashed lines

t = np.linspace(0, t_arrow_length, 1000)

slow_oscillation = 0.6 * np.sin(0.4 * np.pi * t + 0.3) + 0.15 * np.sin(0.9 * np.pi * t)
fast_oscillation = (
    0.2 * np.sin(7 * np.pi * t + 0.1) +
    0.1 * np.sin(11 * np.pi * t + np.pi / 5) +
    0.05 * np.sin(17 * np.pi * t + 1.2)
)
rng = np.random.default_rng(seed=42)
noise = 0.02 * np.convolve(rng.normal(0, 1, len(t)), np.ones(10)/10, mode='same')

# === Riemann integration ===
start, end = t[0], t[-1]
dt0 = 0.05
n_steps = 30

t_riemann = compute_geometric_schedule(dt0, n_steps, end - start)
t_riemann += start
t_riemann = t_riemann[t_riemann <= end]

switch_step = 18
t_switch = t_riemann[switch_step]

# Smooth transition
transition_width = 0.03
transition = 0.5 * (1 - np.tanh((t - t_switch) / transition_width))

# Signal
signal = 1.3 + slow_oscillation + (fast_oscillation + noise) * transition

# Riemann sum (left)
t_leftpoints = t_riemann[:-1]
signal_leftpoints = np.interp(t_leftpoints, t, signal)
areas = signal_leftpoints * np.diff(t_riemann)
riemann_sum = np.sum(areas)

print(f"Approximate integral (Left Riemann sum) = {riemann_sum:.4f}")

# === Helper to draw the coordinate system ===
def draw_coordinate_system(ax, *, with_axis_labels=True, with_markers=True, with_vertical_guides=True,
                           x_tick_positions=None, aspect_equal=True):
    head_len = 0.3
    shaft_len = x_dot_arrow_length - head_len

    # Axes arrows (always)
    ax.arrow(AXIS_X0, 0, t_arrow_length + 1.0, 0,
             head_width=head_len/2, head_length=head_len, fc='black', ec='black')
    ax.arrow(AXIS_X0, 0, 0, shaft_len,
             head_width=head_len/2, head_length=head_len, fc='black', ec='black')

    if with_axis_labels:
        ax.text(t_arrow_length + 1.0, -0.2, r'$t$', va='top', ha='center', fontsize=AXIS_LABEL_FS)
        ax.text(AXIS_X0 - 0.3, x_dot_arrow_length, r'$\dot{x}$', va='bottom', ha='center', fontsize=AXIS_LABEL_FS)

    tick_height = 0.1
    if with_markers:
        # t0, t_switch, tN tick marks + texts
        ax.plot([0, 0], [-tick_height, tick_height], color='black', linewidth=1.0)
        ax.text(0, -0.2, r'$t_0$', va='top', ha='center', fontsize=AXIS_LABEL_FS)

        ax.plot([t_switch, t_switch], [-tick_height, tick_height], color='black', linewidth=1.0)
        ax.text(t_switch, -0.2, r'$t_{\mathrm{switch}}$', va='top', ha='center', fontsize=AXIS_LABEL_FS)

        ax.plot([t[-1], t[-1]], [-tick_height, tick_height], color='black', linewidth=1.0)
        ax.text(t[-1], -0.2, r'$t_{N}$', va='top', ha='center', fontsize=AXIS_LABEL_FS)

    if with_vertical_guides:
        ax.plot([0, 0], [0, dashed_line_height], linestyle='--', color='black', linewidth=1.0)
        ax.plot([t_switch, t_switch], [0, dashed_line_height], linestyle='--', color='black', linewidth=1.0)
        ax.plot([t[-1], t[-1]], [0, dashed_line_height], linestyle='--', color='black', linewidth=1.0)

    # Optional: draw small vertical ticks on the x-axis at specified positions
    if x_tick_positions is not None:
        for xk in x_tick_positions:
            ax.plot([xk, xk], [-tick_height, tick_height], color='black', linewidth=1.0, zorder=3)

    # Final formatting
    ax.set_xlim(AXIS_X0 - 0.3, t_arrow_length + 1.5)
    ax.set_ylim(-0.6, dashed_line_height + 0.5)
    if aspect_equal:
        ax.set_aspect('equal', adjustable='box')
    else:
        ax.set_aspect('auto')
    ax.axis('off')

# === Custom legend handle for "Selected \bar{k}" (L-shape + pentagon) ===
class _SelectedHandle:
    def __init__(self, lw, sel_color, eps_color):
        self.lw = lw
        self.sel_color = sel_color
        self.eps_color = eps_color

class _SelectedHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, x0, y0, w, h, fontsize, trans):
        # horizontal epsilon line (left of marker)
        l_h = Line2D([x0, x0 + w/2], [y0 + h/2, y0 + h/2],
                     linestyle='--', color=orig_handle.eps_color,
                     linewidth=orig_handle.lw, transform=trans, zorder=1)
        # vertical selected-k line (below the marker)
        pad = 0.22
        l_v = Line2D([x0 + w/2, x0 + w/2], [y0 - pad*h, y0 + h/2],
                     linestyle='--', color=orig_handle.sel_color,
                     linewidth=orig_handle.lw, transform=trans, zorder=1, clip_on=False)
        # pentagon marker on top
        m = Line2D([x0 + w/2], [y0 + h/2], marker='p', linestyle='None',
                   markersize=11, markerfacecolor='orange', markeredgecolor='black',
                   transform=trans, zorder=2)
        return [l_h, l_v, m]

# === Plotting ===
if ADD_TUNING_PLOT:
    fig, (ax, ax2) = plt.subplots(
        nrows=2, ncols=1,
        sharex=True, sharey=True,
        figsize=(8, 8),
        gridspec_kw={'hspace': PANEL_HSPACE, 'height_ratios': [1.0, BOTTOM_HEIGHT_RATIO]}
    )
else:
    fig, ax = plt.subplots()

# ---------------- TOP schematic: full coordinate system + content ----------------
draw_coordinate_system(ax, with_axis_labels=True, with_markers=True, with_vertical_guides=True, aspect_equal=True)

# Signal curve (top)
ax.plot(t, signal, lw=1.8, color='black')

# Riemann rectangles (top)
for i in range(len(t_leftpoints)):
    color = 'orange' if i < switch_step else 'lightgreen'
    ax.bar(t_leftpoints[i], signal_leftpoints[i], width=np.diff(t_riemann)[i],
           align='edge', color=color, alpha=0.5, edgecolor='black', linewidth=0.3)

# Panel label (a) — nudged slightly above the axes
ax.text(0.01, 1.04, r'\textbf{(a)}', transform=ax.transAxes,
        ha='left', va='bottom', fontsize=AXIS_LABEL_FS, clip_on=False)

# ---------------- BOTTOM schematic: tuning-style guides + labels & legend ----------------
if ADD_TUNING_PLOT:
    # x-axis ticks aligned with the starts of rectangles above + one extra at the last rectangle end
    x_ticks_bottom = np.append(t_leftpoints, t_riemann[-1])

    # Draw bottom axes with non-equal aspect so the panel can be squashed vertically
    draw_coordinate_system(
        ax2,
        with_axis_labels=False,   # bottom panel uses custom labels
        with_markers=False,
        with_vertical_guides=False,
        x_tick_positions=x_ticks_bottom,
        aspect_equal=False
    )

    # ---------- UNIFIED horizontal log grid (no y-number labels) ----------
    head_len = 0.3
    shaft_len = x_dot_arrow_length - head_len  # y-arrow length
    y_max_grid = shaft_len

    # Choose decades within visible range (start at 0.1 where possible)
    exp_lo = int(np.ceil(np.log10(0.1)))
    exp_hi = int(np.floor(np.log10(y_max_grid)))
    y_decades = 10.0 ** np.arange(exp_lo, exp_hi + 1)

    # Include 2·10^k and 5·10^k as well—style IDENTICAL for all
    y_grid = list(y_decades)
    for e in range(exp_lo, exp_hi + 1):
        y_grid.append(2 * (10.0 ** e))
        y_grid.append(5 * (10.0 ** e))
    y_grid = sorted([yg for yg in y_grid if 0.0 < yg <= y_max_grid])

    # One uniform style for all horizontal grid lines
    for yv in y_grid:
        ax2.hlines(yv, AXIS_X0, t[-1],
                   linestyles='--', linewidth=0.8, color='black', alpha=0.22, zorder=0)

    # Vertical grid lines aligned with switching stages
    for xv in x_ticks_bottom:
        ax2.vlines(xv, 0.0, shaft_len,
                   linestyles='--', linewidth=0.6, color='black', alpha=0.16, zorder=0)
    # ----------------------------------------------------------------------

    # Wiggly decreasing curve (schematic)
    y_start = dashed_line_height * 0.8
    y_end   = 0.20

    x_wig = np.linspace(0.0, t[-1], 800)
    y_lin = y_start + (y_end - y_start) * (x_wig / x_wig[-1])

    # --- WIGGLE CONTROLS ---
    WIGGLE_AMPL  = 1.1
    WIGGLE_FREQ  = 1.5
    SMOOTH_WIN   = 30
    WIGGLE_SEED  = 6
    # -----------------------

    rng_w = np.random.default_rng(seed=WIGGLE_SEED)
    span = abs(y_start - y_end)

    # Sine components (scaled by amplitude and frequency)
    w1 = (0.055 * WIGGLE_AMPL) * span * np.sin(2*np.pi * (0.65 * WIGGLE_FREQ) * (x_wig / x_wig[-1]) + 0.4)
    w2 = (0.030 * WIGGLE_AMPL) * span * np.sin(2*np.pi * (1.85 * WIGGLE_FREQ) * (x_wig / x_wig[-1]) + 1.1)

    # Smooth random component
    noise_w = (0.012 * WIGGLE_AMPL) * span * rng_w.normal(0, 1, x_wig.size)
    kernel = np.ones(int(max(3, round(SMOOTH_WIN)))) / max(3, round(SMOOTH_WIN))
    noise_w = np.convolve(noise_w, kernel, mode='same')

    y_wig = y_lin + w1 + w2 + noise_w
    y_wig = np.clip(y_wig, 0.05, dashed_line_height * 0.95)

    # Plot the wiggly curve
    ax2.plot(x_wig, y_wig, color='tab:blue', linewidth=1.8, zorder=2)

    # === Switching selection guides (schematic) ===
    y_at_switch = np.interp(t_switch, x_wig, y_wig)

    ax2.plot([t_switch, t_switch], [0.0, y_at_switch],
             linestyle='--', linewidth=1.5, color='tab:blue', zorder=1)

    ax2.plot([t_switch], [y_at_switch], marker='p', markersize=9,
             mec='k', mew=1.0, mfc='orange', zorder=3)

    ax2.plot([AXIS_X0, t_switch], [y_at_switch, y_at_switch],
             linestyle='--', linewidth=1.5, color='tab:purple', zorder=1)

    # Purple epsilon label above the horizontal guide
    ax2.annotate(r'$\epsilon$', xy=(AXIS_X0, y_at_switch), xytext=(2, 4),
                 textcoords='offset points', color='tab:purple', ha='left', va='bottom', fontsize=AXIS_LABEL_FS)

    # ===== Bottom axis labels (match top fontsize) =====
    x_mid = AXIS_X0 + 0.5 * (t_arrow_length + 1.0)
    ax2.text(x_mid, -0.18, r'$\mathrm{Switching~stage}$', va='top', ha='center', fontsize=AXIS_LABEL_FS)

    y_mid = 0.5 * shaft_len
    ax2.text(AXIS_X0 - 0.55, y_mid, r'$\mathrm{Cost~incr.~(log)}$',
             rotation=90, va='center', ha='center', fontsize=AXIS_LABEL_FS)

    # ===== Legend: Selected \bar{k} =====
    custom_handle = _SelectedHandle(lw=1.5, sel_color='tab:blue', eps_color='tab:purple')
    legend_kwargs = dict(
        handles=[custom_handle], labels=[r'Selected $\bar{k}$'],
        loc=LEGEND_LOC,
        frameon=True, fancybox=False, framealpha=1.0, edgecolor='black',
        handlelength=1.6, handletextpad=0.28, borderpad=0.28, labelspacing=0.25,
        handler_map={_SelectedHandle: _SelectedHandler()},
        prop={'size': AXIS_LABEL_FS}
    )
    if LEGEND_BBOX is not None:
        legend_kwargs['bbox_to_anchor'] = LEGEND_BBOX
    #ax2.legend(**legend_kwargs)

    # Panel label (b) — nudged slightly above the axes
    ax2.text(0.01, 1.04, r'\textbf{(b)}', transform=ax2.transAxes,
             ha='left', va='bottom', fontsize=AXIS_LABEL_FS, clip_on=False)

plt.show()



