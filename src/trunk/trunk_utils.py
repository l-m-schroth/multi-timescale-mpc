import numpy as np
import matplotlib.pyplot as plt
import mujoco
import os
import mediapy as media
import casadi as ca
from copy import deepcopy
from scipy.optimize import fsolve
from utils_shared import get_dir

def get_x_and_y_pos(angles, link_length):
    # Initialize the starting point (top of the chain)
    x, y = [0], [0]  # Starting at the origin (0, 0)
    
    # Initialize cumulative angle, starting downward
    cumulative_angle = 0 # Downward direction
    
    for angle in angles:
        # Update cumulative angle (relative to the downward direction)
        cumulative_angle += angle
        
        # Calculate next joint position
        x_next = x[-1] + link_length * np.sin(cumulative_angle)
        y_next = y[-1] + -link_length * np.cos(cumulative_angle)
        
        # Append the new joint position
        x.append(x_next)
        y.append(y_next)

    return x, y  


def get_ee_position(data, ee_site_id, steady_state_z_values):
    return data.site_xpos[ee_site_id] - np.array([0, 0, steady_state_z_values[0]])

# Function to reset simulation after initialization
def reset_sim(mjData, qpos, qvel):
    mjData.qpos[:] = qpos
    mjData.qvel[:] = qvel
    mjData.time = 0.0

def get_x_and_y_pos_casadi(angles, link_length):
    # Initialize symbolic variables
    x = [0]
    y = [0]
    cumulative_angle = 0  # Starting downward

    # Use CasADi's vertsplit to iterate over symbolic elements
    for angle in ca.vertsplit(angles):
        cumulative_angle += angle
        x_next = x[-1] + link_length * ca.sin(cumulative_angle)
        y_next = y[-1] + -link_length * ca.cos(cumulative_angle)
        x.append(x_next)
        y.append(y_next)

    return ca.vertcat(*x), ca.vertcat(*y)

# --- function to heuristically map the joint angles to joint angles of lower dimensional chains --- #
def compute_q(q_high_dim, link_length_high_dim, link_length_low_dim, n_chains_low):
    n_chains_high = len(q_high_dim)
    x_high, y_high = get_x_and_y_pos(q_high_dim, link_length_high_dim)
    step_size = int(n_chains_high/n_chains_low)
    q_low = []
    q_tot, x_actual, y_actual = 0, 0, 0
    for i in range(n_chains_low):
        x_diff = x_high[step_size*(i+1)] - x_actual
        y_diff = y_high[step_size*(i+1)] - y_actual
        q_low_next = np.arctan2(x_diff, -y_diff) - q_tot
        q_low.append(q_low_next)
        q_tot += q_low_next
        x_actual += np.sin(q_tot)*link_length_low_dim
        y_actual += -np.cos(q_tot)*link_length_low_dim
    return q_low

def compute_q_casadi(q_high_dim, link_length_high_dim, link_length_low_dim, n_chains_low):
    # Get dimensions
    n_chains_high = q_high_dim.size()[0]  # Size in CasADi syntax
    step_size = n_chains_high // n_chains_low

    # Symbolic variables
    x_high, y_high = get_x_and_y_pos_casadi(q_high_dim, link_length_high_dim)
    
    # Initialize low-dimensional chain computation
    q_low = []
    q_tot = 0
    x_actual, y_actual = 0, 0
    
    for i in range(n_chains_low):
        x_diff = x_high[step_size * (i + 1)] - x_actual
        y_diff = y_high[step_size * (i + 1)] - y_actual
        q_low_next = ca.arctan2(x_diff, -y_diff) - q_tot
        q_low.append(q_low_next)
        q_tot += q_low_next
        x_actual += ca.sin(q_tot) * link_length_low_dim
        y_actual += -ca.cos(q_tot) * link_length_low_dim

    return ca.vertcat(*q_low)  # Return as CasADi symbolic vector

def simulate(mjModel, mjData, duration, ee_site_id, qpos_init, qvel_init, create_video=False, u_controls = None):

    # reset sim to inital condition
    reset_sim(mjData, qpos_init, qvel_init)

    # init lists
    mujoco.mj_fwdPosition(mjModel, mjData)
    ee_positions = []
    q_positions = [qpos_init] 
    q_velocities = [qvel_init] 
    u_counter = 0
    if u_controls is not None:
        if isinstance(u_controls, list):
            u_controls = np.array(u_controls)
        if len(u_controls.shape) > 1:
            multiple_u = True
        else:
            multiple_u = False
    if create_video:
        frames = []
        framerate = 60

    with mujoco.Renderer(mjModel) as renderer:
        while mjData.time <= duration - mjModel.opt.timestep:

            if u_controls is not None:
                middle = int(len(mjData.ctrl[:])/2)
                if multiple_u: 
                    u = u_controls[u_counter, :]
                else:
                    u = u_controls    
                mjData.ctrl[:middle] = u[0]
                mjData.ctrl[middle:] = u[1]
                u_counter += 1

            mujoco.mj_step(mjModel, mjData)
            
            ee_pos = get_ee_position(mjData, ee_site_id, [0.2]) # ee steady state z_value is 0.2
            ee_positions.append(ee_pos)

            # Render the frame
            if create_video and len(frames) < mjData.time * framerate:
                renderer.update_scene(mjData)
                pixels = renderer.render()
                frames.append(pixels)

            #print(mjData.qpos)
            #print(mjData.qvel)
            q_positions.append(deepcopy(mjData.qpos))
            q_velocities.append(deepcopy(mjData.qvel))

    # Save rendered video
    if create_video:
        current_dir = os.path.dirname(__file__)
        video_folder = os.path.join(current_dir, "trajectories", "videos")
        if u_controls is not None:
            output_video_file = os.path.join(video_folder, f"video_chain_comparison_{len(qpos_init)}_controlled.mp4")
        else:    
            output_video_file = os.path.join(video_folder, f"video_chain_comparison_{len(qpos_init)}.mp4")
        media.write_video(output_video_file, frames, fps=framerate)
        print(f"Simulation video saved to {output_video_file}.")

    return ee_positions, q_positions, q_velocities
