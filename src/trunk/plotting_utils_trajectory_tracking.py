from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import mujoco
from plotting_utils_shared import latexify_plot
from utils_shared import get_dir
from trunk.trunk_utils import get_ee_position

def plot_closed_loop_trajectory(frequency, Phi_t_func, ee_pos, save=False, number=-1, latexify=False):
    
    if latexify:
        latexify_plot(fontsize=16)

    # evaluate one period for plotting
    T = 1 / frequency  # Period of the motion
    time_steps = np.linspace(0, T, 100)  # 100 time steps within one period

    # Evaluate the trajectory using Phi_t_func
    trajectory = np.array([Phi_t_func(t_val) for t_val in time_steps])

    # Extract x and z coordinates
    trajectory_x = trajectory[:, 0]  # x-coordinates
    trajectory_z = trajectory[:, 1]  # z-coordinates

    # Convert endeffector position to numpy array for plotting
    ee_pos = np.array(ee_pos)

    x_positions = ee_pos[:, 0]
    z_positions = ee_pos[:, 1] 

    # Plot the development of the end-effector in the x-z plane
    plt.plot(x_positions, z_positions , color='red', label='Closed-loop Trajectory')
    plt.plot([0.2, 0.2], [-0.5, -0.25], color='black', linestyle='--', linewidth=2.0, label='Constraint')

    # Add an ellipsoid with parameters a and b around the origin
    plt.plot(trajectory_x, trajectory_z, '--', color='green', label='Desired Trajectory')

    plt.xlabel('x-position [m]')
    plt.ylabel('z-position [m]')
    #plt.title('End-Effector Trajectory in x-z Plane')
    plt.axis('equal')  # Make the axes equal
    plt.xlim([-0.3, 0.3])  # Set the range from -1.0 to 2.0 on the x-axis
    plt.legend(loc="upper left")
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5)
    if save:
        trunk_plots_dir = get_dir("plots/trunk")
        filename_pdf = trunk_plots_dir / f"closed_loop_trajectory_{number}.pdf"
        plt.savefig(filename_pdf, bbox_inches="tight")
        filename_pgf = trunk_plots_dir / f"closed_loop_trajectory_{number}.pgf"
        plt.savefig(filename_pgf, bbox_inches="tight")
    plt.show()

def plot_closed_loop_trajectories_jointly(frequency, dt_sim, Phi_t_func, ee_pos_baseline, ee_pos_6, save=False, latexify=False):
    if latexify:
        latexify_plot(fontsize=21)

    T = 1.0 / frequency
    t_grid = np.linspace(0.0, T, 100)
    ref_traj = np.asarray([Phi_t_func(t) for t in t_grid])
    ref_x, ref_z = ref_traj[:, 0], ref_traj[:, 1]

    ee_pos_baseline = np.asarray(ee_pos_baseline)
    ee_pos_6 = np.asarray(ee_pos_6)

    n_per_iteration = int(frequency/ dt_sim)

    plt.figure(figsize=(6, 4.5))  # width, height in inches
    plt.plot(ee_pos_6[n_per_iteration:2*n_per_iteration, 0], ee_pos_6[n_per_iteration:2*n_per_iteration, 1], color='orange', linewidth=3.0, label='6) Proposed approach')
    plt.plot(ee_pos_baseline[n_per_iteration:2*n_per_iteration, 0], ee_pos_baseline[n_per_iteration:2*n_per_iteration, 1], color='blue', linestyle='--', linewidth=3.0, label='0) Baseline')
    plt.plot([0.2, 0.2], [-0.5, -0.25], color='black', linestyle=':', linewidth=3.0, label='Constraint')
    plt.plot(ref_x, ref_z, color='green', linestyle='--', label='Desired Trajectory', linewidth=3.0)

    plt.xlabel('x-position [m]')
    plt.ylabel('z-position [m]')
    plt.axis('equal')
    plt.xlim([-0.30, 0.30])
    plt.ylim([-0.55, -0.05])
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5)
    #plt.legend(loc='upper left')

    if save:
        trunk_plots_dir = get_dir("plots/trunk")
        for ext in ("pdf", "pgf"):
            plt.savefig(trunk_plots_dir / f"closed_loop_trajectory_jointly.{ext}", bbox_inches="tight")

    plt.show()


def plot_open_loop_plan_ee(trunkMPC, circle_x, circle_z, n_high, ee_side_id, ground_truth_mujoco=True, mjData=None, mjModel=None):
    # Extract planned x and z positions
    traj_x_ee = trunkMPC.get_planned_ee_trajectory()
    planned_x_positions = traj_x_ee[:, 0]
    planned_z_positions = traj_x_ee[:, 1]

    # Create a new plot
    plt.figure()
    plt.scatter(planned_x_positions[0], planned_z_positions[0], label='Start Point', color='red')
    plt.plot(planned_x_positions, planned_z_positions, label='Planned Trajectory', color='blue')
    plt.plot(circle_x, circle_z, '--', label='Reference Trajectory', color='green')  # Reference circle for comparison
    # Plot x-axis constraints (vertical lines)
    if trunkMPC.ub_x_ee[0] < 1e15:
        plt.axvline(x=trunkMPC.ub_x_ee[0], color='r', linestyle='--', linewidth=2, label="ub_x")
    if trunkMPC.lb_x_ee[0] > -1e15:
        plt.axvline(x=trunkMPC.lb_x_ee[0], color='r', linestyle='--', linewidth=2, label="lb_x")

    # Plot y-axis constraints (horizontal lines)
    if trunkMPC.ub_x_ee[1] < 1e15:
        plt.axhline(y=trunkMPC.ub_x_ee[1], color='b', linestyle='--', linewidth=2, label="ub_y")
    if trunkMPC.lb_x_ee[1] > -1e15:
        plt.axhline(y=trunkMPC.lb_x_ee[1], color='b', linestyle='--', linewidth=2, label="lb_y")

    if ground_truth_mujoco:

        acados_offset = -0.5

        # Deepcopy mjData and mjModel
        mjData_copy = deepcopy(mjData)
        mjModel_copy = deepcopy(mjModel)

        # Initialize ground truth trajectory
        ground_truth_x = []
        ground_truth_z = []

        steady_state_z_ee_acados_coordinates = [0.2 - acados_offset]
        mujoco.mj_forward(mjModel_copy, mjData_copy)
        # ee_pos = get_ee_position(mjData_copy, ee_side_id, steady_state_z_ee_acados_coordinates)
        # ground_truth_x.append(ee_pos[0]) 
        # ground_truth_z.append(ee_pos[2])  

        # Simulate the trajectory
        if isinstance(trunkMPC.n,list) and trunkMPC.n[-1] == 'p':
            ground_truth_steps = len(planned_x_positions[:-trunkMPC.N_list[-1]]) - trunkMPC.n_phases
        else:
            ground_truth_steps = len(planned_x_positions) - trunkMPC.n_phases
        for i in range(ground_truth_steps):
            # Record the current end-effector position
            ee_pos = get_ee_position(mjData_copy, ee_side_id, steady_state_z_ee_acados_coordinates)
            ground_truth_x.append(ee_pos[0])  
            ground_truth_z.append(ee_pos[2])  

            # Apply control and simulate forward
            control_input = trunkMPC.acados_ocp_solver.get(i, "u")

            # time step 
            time_step = trunkMPC.dt[i] 
            if control_input.size > 0:
                mjData_copy.ctrl[:] = np.array([control_input[0]] * (n_high // 2) + [control_input[1]] * (n_high // 2))
                for _ in range(int(time_step / mjModel_copy.opt.timestep)):
                    # Step simulation
                    mujoco.mj_step(mjModel_copy, mjData_copy)

        # Plot ground truth trajectory
        plt.plot(ground_truth_x, ground_truth_z, label='Ground Truth Trajectory', color='orange')
    
    # Plot details
    plt.xlabel('x position (m)')
    plt.ylabel('z position (m)')
    plt.title('Planned Trajectory vs Reference Circle')
    plt.axis('equal')  # Make the axes equal
    plt.legend()
    plt.grid()
    plt.show()
