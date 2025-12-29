from pathlib import Path 
import matplotlib.pyplot as plt
from scipy.optimize import bisect
import numpy as np

def get_dir(folder="plots"):
    repo_root = Path(__file__).resolve().parents[1]
    plots_dir = repo_root / folder
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir

def compute_exponential_step_sizes(dt_initial, T_total, N_steps, plot=False):
    """
    Compute exponentially increasing step sizes such that their sum equals T_total.

    Args:
        dt_initial (float): Initial step size (e.g. 0.002).
        T_total (float): Total time horizon (e.g. 20.0).
        N_steps (int): Number of steps (e.g. 100).
        plot (bool): Whether to plot the step sizes (default: False).

    Returns:
        np.ndarray: Array of step sizes of length N_steps.
    """

    # Function to find r: the common ratio of the geometric series
    def geometric_sum_error(r):
        if np.isclose(r, 1.0):
            return dt_initial * N_steps - T_total
        return dt_initial * (1 - r**N_steps) / (1 - r) - T_total

    # Find r using root-finding
    r = bisect(geometric_sum_error, 0.9, 5.0)

    # Generate step sizes
    step_sizes = np.array([dt_initial * r**i for i in range(N_steps)])

    # Optional plot
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(range(N_steps), step_sizes, marker='o')
        plt.xlabel("Step Index")
        plt.ylabel("Step Size")
        plt.title("Exponential Step Size Growth")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return step_sizes