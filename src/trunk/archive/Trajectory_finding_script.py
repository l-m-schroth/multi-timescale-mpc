"""
Functionality that allows for easy generation of reference trajectory by fitting a spline through selected points
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from trunk.trunk_utils import get_x_and_y_pos
import os
import pickle
from utils_shared import get_dir

def select_periodic_trajectory(trajectory_name, n=8, frequency=1.0, link_length_total=0.5, num_samples=20, plot=True):
    """
    Explore reachable positions, interactively define points, and compute or load a periodic trajectory function and its derivative.

    Args:
        trajectory_name (str): Name of the trajectory file to save/load.
        n (int): Number of links in the chain.
        link_length_total (float): Total length of the chain (default is 0.5).
        num_samples (int): Number of samples per angle (default is 50).

    Returns:
        tuple: (trajectory_func, derivative_func)
            - trajectory_func: A function that computes [x, y] coordinates for a given time t.
            - derivative_func: A function that computes [dx/dt, dy/dt] for a given time t.
    """
    
    trunk_dir = get_dir("src/trunk")
    trajectory_file = os.path.join(trunk_dir, f"trajectories_ee_tracking/{trajectory_name}.pkl")

    if os.path.exists(trajectory_file):
        # Load the trajectory from the file
        print(f"Loading trajectory from {trajectory_file}...")
        with open(trajectory_file, 'rb') as f:
            tck = pickle.load(f)
    else:
        # Compute the length of a single link
        link_length = link_length_total / n

        # Create an array of angles for the first half of the chain
        angle_range = np.linspace(-np.pi / 16, np.pi / 16, num_samples)

        # Prepare a grid of angles for the first half of the chain
        angle_combinations = np.meshgrid(*[angle_range] * (n // 2))
        angle_combinations = np.stack(angle_combinations, axis=-1).reshape(-1, n // 2)

        # Prepare an array to store the end-effector positions
        ee_positions = []

        # Loop through each angle combination and calculate the end-effector position
        for angles_first_half in angle_combinations:
            angles = np.concatenate([angles_first_half, np.zeros(n // 2)])
            x_positions, y_positions = get_x_and_y_pos(angles, link_length)
            ee_positions.append([x_positions[-1], y_positions[-1]])

        ee_positions = np.array(ee_positions)

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(ee_positions[:, 0], ee_positions[:, 1], s=1, label="Reachable Positions")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title(f"Define a Periodic Trajectory (n={n})")
        ax.axis("equal")
        ax.grid(True)
        ax.legend()

        # List to store user-clicked points
        selected_points = []

        def onclick(event):
            if event.inaxes == ax:
                x, y = event.xdata, event.ydata
                selected_points.append([x, y])
                ax.plot(x, y, 'ro')
                fig.canvas.draw()

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        fig.canvas.mpl_disconnect(cid)

        if len(selected_points) < 3:
            print("At least three points are required to compute a closed trajectory.")
            return

        selected_points = np.array(selected_points)
        selected_points = np.vstack([selected_points, selected_points[0]])
        selected_points = selected_points.T

        # Compute a smooth closed trajectory through the selected points
        tck, _ = splprep(selected_points, per=True, s=0)

        # Save the trajectory to the file
        with open(trajectory_file, 'wb') as f:
            pickle.dump(tck, f)
        print(f"Trajectory saved as {trajectory_file}.")

    # Create a periodic trajectory function
    def trajectory_func(t):
        t_normalized = (t * frequency) % 1.0
        x, y = splev(t_normalized, tck)
        return [x, y]

    # # Compute finite differences for the derivative
    # def derivative_func(t):
    #     t_normalized = (t * frequency) % 1.0
    #     dx_dt, dy_dt = splev(t_normalized, tck, der=1)
    #     return [dx_dt, dy_dt]
    
    def derivative_func(t):
        t_normalized = (t * frequency) % 1.0
        dx_dt, dy_dt = splev(t_normalized, tck, der=1)
        return [dx_dt * frequency, dy_dt * frequency]  # Scale derivatives correctly


    # Evaluate and plot the trajectory
    times = np.linspace(0, 1 / frequency, 300)
    trajectory_points = np.array([trajectory_func(t) for t in times])
    derivative_points = np.array([derivative_func(t) for t in times])
    
    if plot:
        plt.figure(figsize=(8, 8))
        plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], 'b-', label="Periodic Trajectory")
        plt.quiver(
            trajectory_points[:, 0], trajectory_points[:, 1],
            derivative_points[:, 0], derivative_points[:, 1],
            angles='xy', scale_units='xy', scale=1, color='orange', label="Derivative (dx/dt, dy/dt)"
        )
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Periodic Trajectory and Derivative")
        plt.axis("equal")
        plt.legend()
        plt.grid(True)
        plt.show()

    return trajectory_func, derivative_func

def main():
    # Hardcoded parameters matching the Jupyter notebook setup
    trajectory_name = "trajectory_new" # Choose not existing name to generate new trajectory
    n = 8
    frequency = 1.0
    link_length_total = 0.5
    num_samples = 20
    plot = True  # enable interactive point selection

    # Call the existing function to generate (or load) and save the trajectory
    select_periodic_trajectory(
        trajectory_name=trajectory_name,
        n=n,
        frequency=frequency,
        link_length_total=link_length_total,
        num_samples=num_samples,
        plot=plot
    )


if __name__ == "__main__":
    main()



