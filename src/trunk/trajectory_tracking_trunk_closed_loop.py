"""
Closed-loop simulation functionality
"""
import mujoco
import numpy as np
from trunk.trunk_utils import compute_q
from trunk.trunk_utils import get_ee_position
from trunk.plotting_utils_trajectory_tracking import plot_open_loop_plan_ee
import os
import mediapy as media
import casadi as ca
from trunk.trunk_utils import compute_q_casadi
from trunk.ODE_chains.ODE_utils import get_ode_params_dict
from utils_shared import get_dir

def closed_loop_ee_tracking_mujoco(n, n_high, Mjmodel, Mjdata, duration, framerate, create_video, plot_open_loop_plan_bool, trajectory_x, trajectory_z, trunkMPC, controller_interval=None, perturbation_variance=0.0):
    # functions for mapping the states between chains of different dimensions, in case we try to control a higher dimensional chain with a lower dimensional model, otherwhise the mappings are the identity
    q_high_dim = ca.SX.sym(f'q_high_dim_{n_high}', n_high)  # Symbolic q_high for 16-link
    q_dot_high_dim = ca.SX.sym(f'q_dot_high_dim_{n_high}', n_high)  # Symbolic q_high_dot for 16-link

    ode_params_dict = get_ode_params_dict()
    link_length_high_dim = 2*ode_params_dict[f"{n_high}"]["l"]
    link_length_low_dim = 2*ode_params_dict[f"{n[0]}"]["l"]
    q_low_dim = compute_q_casadi(q_high_dim, link_length_high_dim, link_length_low_dim, n[0])
    jacobian_q_low_q_high = ca.jacobian(q_low_dim, q_high_dim)  # Jacobian of q_8 w.r.t. q_16
    q_dot_low_dim = jacobian_q_low_q_high @ q_dot_high_dim  # Velocity mapping for 8-link
    q_dot_low_dim_fn = ca.Function('q_dot_low_dim_fn', [q_high_dim, q_dot_high_dim], [q_dot_low_dim])  # CasADi function

    # set control frequency to the timestep of the MPC
    if controller_interval is not None:
        controller_interval = controller_interval
    else:
        controller_interval = trunkMPC.dt
    
    # simulate
    ee_side_id = mujoco.mj_name2id(Mjmodel, mujoco.mjtObj.mjOBJ_SITE, "endeffector")
    frames = []
    control_inputs = []
    ee_pos = []
    costs = []
    solve_times = []
    SQP_iters = []
    constraint_violations = 0
    last_update_time = -controller_interval  # Initialize to ensure the controller is applied at the start
    with mujoco.Renderer(Mjmodel, width=1920, height=1080) as renderer:
        while Mjdata.time < duration:
            # extract and save endeffector position
            mujoco.mj_forward(Mjmodel, Mjdata) # just to be save that all positions etc are updated
            #print(f"MuJoCo Timestep: {Mjmodel.opt.timestep}")

            ee_pos.append(get_ee_position(Mjdata, ee_side_id, [0.7]))

            x_ee, _, y_ee = ee_pos[-1]  # Assuming ee_pos is a list of (x, y) positions

            # Check violations for x position
            if x_ee > trunkMPC.ub_x_ee[0]:  
                constraint_violations += (x_ee - trunkMPC.ub_x_ee[0])
            if x_ee < trunkMPC.lb_x_ee[0]:  
                constraint_violations += (trunkMPC.lb_x_ee[0] - x_ee)

            # Check violations for y position
            if y_ee > trunkMPC.ub_x_ee[1]:  
                constraint_violations += (y_ee - trunkMPC.ub_x_ee[1])
            if y_ee < trunkMPC.lb_x_ee[1]:  
                constraint_violations += (trunkMPC.lb_x_ee[1] - y_ee)

            # Apply controller at given frequency
            if Mjdata.time - last_update_time >= controller_interval:
                x0 = np.concatenate((Mjdata.qpos, Mjdata.qvel))
                q0_high_dim = Mjdata.qpos
                q0_dot_high_dim = Mjdata.qvel
                q0_mapped = compute_q(q0_high_dim, link_length_high_dim, link_length_low_dim, n[0])
                q0_dot_mapped = q_dot_low_dim_fn(q0_high_dim, q0_dot_high_dim).full().flatten()
                x0_mapped = np.concatenate((q0_mapped, q0_dot_mapped))
                trunkMPC.set_initial_state(x0_mapped)
                trunkMPC.set_reference_trajectory(Mjdata.time)
                u0, solve_time_tot, sqp_iters = trunkMPC.solve_ocp()
                step = int(round(Mjdata.time / trunkMPC.dt[0]))
                if step % 200 == 0 and plot_open_loop_plan_bool:
                    plot_open_loop_plan_ee(trunkMPC, trajectory_x, trajectory_z, n_high, ee_side_id, ground_truth_mujoco=True, mjData=Mjdata, mjModel=Mjmodel)
                u0_mapped = u0
                Mjdata.ctrl[:] = np.array([u0_mapped[0]] * (n_high // 2) + [u0_mapped[1]] * (n_high // 2))
                last_update_time = Mjdata.time

                # append solve time
                solve_times.append(solve_time_tot)
                SQP_iters.append(sqp_iters)
            
            # compute costs
            if trunkMPC.multi_phase:
                Q, R = trunkMPC.ocp.cost[0].Q, trunkMPC.ocp.cost[0].R
            else:
                Q, R = trunkMPC.ocp.cost.Q, trunkMPC.ocp.cost.R
            ee_pos_diff = np.delete(ee_pos[-1], 1) - trunkMPC.Phi_t(Mjdata.time)
            site_velocity = np.zeros(6)
            mujoco.mj_objectVelocity(Mjmodel, Mjdata, mujoco.mjtObj.mjOBJ_SITE, ee_side_id, site_velocity, flg_local=False)
            ee_site_linear_velocity = site_velocity[3:]
            ee_dot_diff = np.delete(ee_site_linear_velocity,1) - trunkMPC.Phi_dot_t(Mjdata.time)
            y_diff = np.concatenate((ee_pos_diff, ee_dot_diff))
            costs.append(y_diff.T @ Q @ y_diff + u0_mapped.T @ R @  u0_mapped)

            mujoco.mj_step(Mjmodel, Mjdata)
            # add noise to the state 
            # Add random noise to joint velocities (or positions, if needed)
            noise = np.random.normal(0.000, np.sqrt(perturbation_variance), Mjdata.qpos.shape)
            # Mjdata.qvel += noise  # Apply noise
            # Mjdata.qpos += noise  # Apply noise to positions
            # try if applying a force at the endeffector produces more predictable results
            endeffector_body_id = Mjmodel.site_bodyid[ee_side_id]
            sigma = np.sqrt(perturbation_variance)
            force_2d = np.random.normal(0, sigma, size=2)
            force_x, force_z = force_2d
            Mjdata.xfrc_applied[endeffector_body_id, 0] = force_x
            Mjdata.xfrc_applied[endeffector_body_id, 1] = 0.0
            Mjdata.xfrc_applied[endeffector_body_id, 2] = force_z

            control_inputs.append(u0_mapped)

            if create_video and len(frames) < Mjdata.time * framerate:
                renderer.update_scene(Mjdata, camera="yz_view")
                pixels = renderer.render()
                frames.append(pixels)

    if create_video:
        plots_dir = get_dir("plots/trunk")
        video_folder = os.path.join(plots_dir, f"reference_tracking_2d_trunk_{n_high}_{n}.mp4")
        media.write_video(video_folder, frames, fps=framerate)
        print('Rendering done')

    return control_inputs, ee_pos, costs, solve_times, SQP_iters, constraint_violations


def closed_loop_ee_tracking_acados(n, n_high, duration, plot_open_loop_plan_bool, trajectory_x, trajectory_z, trunkMPC, sim_solver=None):  
    "closed loop sim in acados"
    
    control_inputs = []
    ee_pos = []
    costs = []
    solve_times = []
    SQP_iters = []
    constraint_violations = 0
    time = 0
    
    x0 = np.zeros((2 * n[0],))

    states = []
    time_list = []

    # Prepare EE-velocity function 
    q = ca.MX.sym('q', n[0])  # generalized coodinates
    q_dot = ca.MX.sym('q_dot', n[0])  # generalized coodinates
    
    # map high dimensional q and q_dots to endeffector position
    x_pos_ee, y_pos_ee = trunkMPC.forward_kinematics_casadi(q, n[0])   
    x_ee_dot = ca.jacobian(x_pos_ee, q) @ q_dot 
    y_ee_dot = ca.jacobian(y_pos_ee, q) @ q_dot   

    EE_dot_fct = ca.Function('ee_dot', [q, q_dot], [ca.vertcat(x_ee_dot, y_ee_dot)])

    if not sim_solver:
        sim_solver = trunkMPC.acados_sim_solver
    control_step = int(trunkMPC.dt[0]/sim_solver.T)
    switched = False
    num_steps = int(duration / sim_solver.T)
    for step in range(num_steps):
        
        if step % control_step == 0:
            trunkMPC.set_reference_trajectory(time)
            trunkMPC.set_initial_state(x0)
            u0, solve_time_tot, sqp_iters = trunkMPC.solve_ocp()
            solve_times.append(solve_time_tot)
            SQP_iters.append(sqp_iters)

        ee_pos.append(trunkMPC.forward_kinematics_casadi(x0[:n[0]], n[0]))
        # Extract end-effector position
        x_ee, y_ee = ee_pos[-1]  # Assuming ee_pos is a list of (x, y) positions

        # Check violations for x position
        if x_ee > trunkMPC.ub_x_ee[0]:  
            constraint_violations += (x_ee - trunkMPC.ub_x_ee[0])
        if x_ee < trunkMPC.lb_x_ee[0]:  
            constraint_violations += (trunkMPC.lb_x_ee[0] - x_ee)

        # Check violations for y position
        if y_ee > trunkMPC.ub_x_ee[1]:  
            constraint_violations += (y_ee - trunkMPC.ub_x_ee[1])
        if y_ee < trunkMPC.lb_x_ee[1]:  
            constraint_violations += (trunkMPC.lb_x_ee[1] - y_ee)
        
        sim_solver.set("x", x0)
        sim_solver.set("u", u0)
        status = sim_solver.solve()
        if status != 0:
            raise RuntimeError(f"ACADOS sim solver failed with status {status}.")
        
        if step % 250 == 0 and plot_open_loop_plan_bool:
            print("computation time:", solve_time_tot)
            plot_open_loop_plan_ee(trunkMPC, trajectory_x, trajectory_z, n_high, None, ground_truth_mujoco=False)

        # compute costs
        if trunkMPC.multi_phase:
            Q, R = trunkMPC.ocp.cost[0].Q, trunkMPC.ocp.cost[0].R
        else:
            Q, R = trunkMPC.ocp.cost.Q, trunkMPC.ocp.cost.R
        ee_pos_diff = np.array(ee_pos[-1]) - trunkMPC.Phi_t(time)
        ee_linear_velocity = EE_dot_fct(x0[:n[0]], x0[-n[0]:]).full().squeeze()
        #test = trunkMPC.Phi_dot_t(time)
        ee_dot_diff = ee_linear_velocity - trunkMPC.Phi_dot_t(time)
        y_diff = np.concatenate((ee_pos_diff, ee_dot_diff))
        costs.append(y_diff.T @ Q @ y_diff + u0.T @ R @  u0)
        
        x_next = sim_solver.get("x")
        states.append(x0)
        x0 = x_next
        time_list.append(sim_solver.T)
        time += sim_solver.T

        control_inputs.append(u0)

    return control_inputs, states, time_list, ee_pos, costs, solve_times, SQP_iters, constraint_violations
    
