import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from plotting_utils_shared import latexify_plot
from utils_shared import get_dir

def plot_diff_drive_trajectory(y_ref, mpc=None, closed_loop_traj=None, open_loop_plan=None, 
                               plot_errors=False, latexify=False, number=None, legend=False, 
                               save=False, initial_state=None, closed_loop_traj_baseline=None):
    """
    Plots the 2D trajectory of the differential drive robot with additional features.
    
    Args:
        y_ref (np.ndarray): Target state [x_ref, y_ref, theta_ref].
        mpc (object, optional): MPC instance
        closed_loop_traj (np.ndarray, optional): Closed-loop trajectory of shape (time_steps, >=4).
        open_loop_plan (list or np.ndarray, optional): Open-loop MPC plan of shape (time_steps, >=4).
        plot_errors (bool, optional): If True, plots orientation arrows along the trajectory.
        latexify (bool, optional): If True, applies latexified styles for plotting.
        number (int, optional): If given, plots the number in the lower-left corner.
        legend (bool, optional): If True, displays the legend.
        save (bool, optional): If True, saves the figure as "plots/diff_drive_trajectory_{number}.pgf".
        initial_state (np.ndarray, optional): Initial state [x0, y0, theta0] to be plotted.
    """
    if latexify:
        latexify_plot(fontsize=30)  # Assumes you have a latexify_plot() defined elsewhere

    plt.figure(figsize=(8, 6))
    
    # Ensure y_ref is a 1D array and extract scalar values
    y_ref = np.array(y_ref).ravel()
    # Plot the target state as an orange dot with an orientation arrow
    plt.scatter(y_ref[0], y_ref[1], color='orange', marker='o', s=150, label="Goal state")
    dx_ref = 0.1 * np.cos(y_ref[2])
    dy_ref = 0.1 * np.sin(y_ref[2])
    plt.arrow(y_ref[0], y_ref[1], dx_ref, dy_ref, head_width=0.04, color='orange', linewidth=3.0)
    
    # Plot the initial state with its orientation arrow if provided
    if initial_state is not None:
        init_state = np.array(initial_state).ravel()
        if init_state.shape[0] != 3:
            raise ValueError(f"initial_state must be of length 3, got shape {init_state.shape}")
        x0, y0, theta0 = init_state  # Extract scalar values
        plt.scatter(x0, y0, color='grey', marker='o', s=150, label="Initial state")
        dx_init = 0.1 * np.cos(theta0)
        dy_init = 0.1 * np.sin(theta0)
        plt.arrow(x0, y0, dx_init, dy_init, head_width=0.04, color='grey', linewidth=3.0)
    
    # Plot closed-loop trajectory if provided
    if closed_loop_traj is not None:
        x_closed, y_closed = closed_loop_traj[:, 0], closed_loop_traj[:, 1]
        plt.plot(x_closed, y_closed, '-', color='orange', label="6) FFMPC (Ours)", linewidth=3)
        
        # Add orientation arrows along the trajectory if requested
        if plot_errors:
            for i in range(0, len(x_closed), max(1, len(x_closed) // 10)):
                theta = closed_loop_traj[i, 3]  # Assuming theta is stored at index 3
                dx, dy = np.cos(theta) * 0.1, np.sin(theta) * 0.1
                plt.arrow(x_closed[i], y_closed[i], dx, dy, head_width=0.03, color='r')

    # Plot closed-loop trajectory of baseline if provided
    if closed_loop_traj_baseline is not None:
        x_closed_base, y_closed_base = closed_loop_traj_baseline[:, 0], closed_loop_traj_baseline[:, 1]
        plt.plot(x_closed_base, y_closed_base, 'b--', label="0) Baseline", linewidth=3)
        
    
    # Plot open-loop trajectory if provided
    if open_loop_plan is not None:
        x_open = np.array([arr.squeeze()[0] for arr in open_loop_plan])
        y_open = np.array([arr.squeeze()[1] for arr in open_loop_plan])
        plt.plot(x_open, y_open, 'b--', label="Open Loop Plan", linewidth=2)
        
        if plot_errors:
            for i in range(0, len(x_open), max(1, len(x_open) // 10)):
                theta = open_loop_plan[i][3]  # Assuming theta is stored at index 3
                dx, dy = np.cos(theta) * 0.1, np.sin(theta) * 0.1
                plt.arrow(x_open[i], y_open[i], dx, dy, head_width=0.03, color='b')
    
    # Add the number in the lower-left corner if provided
    if number is not None:
        plt.text(
            0.05, 0.05,  # Coordinates for lower-left corner
            f"{number})",
            transform=plt.gca().transAxes,
            horizontalalignment='left',
            verticalalignment='bottom',
            fontsize=32,
            bbox=dict(facecolor='white', alpha=1.0, edgecolor='none', pad=8)
        )
    
    # Labels and settings
    plt.xlabel("x-position [m]")
    plt.ylabel("y-position [m]")
    #plt.title("Differential Drive Robot Trajectory")
    
    if legend:
        legend_fontsize = 30
        plt.legend(loc='upper right', fontsize=legend_fontsize, framealpha=1.0)
    
    plt.grid()
    plt.axis("equal")  # Ensures equal scaling for x and y axes
    
    # Save figure if requested
    if save:
        plots_dir = get_dir("plots")
        diff_plots_dir = plots_dir / "differential_drive"
        diff_plots_dir.mkdir(parents=True, exist_ok=True)

        if number is not None:
            filename = diff_plots_dir / f"diff_drive_trajectory_{number}.pdf"
        else:
            filename = diff_plots_dir / "diff_drive_trajectory.pdf"

        plt.savefig(filename, bbox_inches="tight")
    
    plt.show()

