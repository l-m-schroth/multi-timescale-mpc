import numpy as np

def simulate_closed_loop_drone(
    x0, 
    mpc, 
    duration, 
    stage_cost_function,
    target_xy,
    sigma_noise=0.0, 
    tau_noise=1.0, 
    sim_solver=None, 
    control_step=1,
):
    """
    Simulate a closed-loop system for DroneMPC with zero-order hold (ZOH) control updates,
    supporting Ornstein-Uhlenbeck noise.

    Parameters
    ----------
    x0 : np.ndarray
        Initial state of dimension matching the sim solver (e.g. 12D for full model, 10D ignoring pendulum, etc.).
    mpc : DroneMPC
        The DroneMPC instance with .solve(...) method and .opts containing relevant Q, R, etc.
    duration : float
        Total simulation time in seconds.
    sigma_noise : float, optional
        Standard deviation of the OU process noise.
    tau_noise : float, optional
        Time constant controlling the OU noise correlation. Lower => higher frequency.
    sim_solver : AcadosSimSolver, optional
        If not provided, use e.g. mpc.acados_sim_solver from the DroneMPC class.
    control_step : int, optional
        Number of sim steps between MPC updates (ZOH in between).

    Returns
    -------
    x_traj : np.ndarray
        State trajectory of shape (num_steps+1, x_dim).
    u_traj : np.ndarray
        Applied controls of shape (num_steps, u_dim).
    stage_costs : list of float
        List of scalar stage costs at each step.
    solve_times : list of float
        Time taken by each MPC solve (for profiling).
    number of iters : 
        List of scalar number of SQP-iterations.
    """

    # If no external sim_solver is provided, use the drone's default
    if sim_solver is None:
        # For example, if your DroneMPC created a sim solver for the full model:
        sim_solver = mpc.sim_solver_full if hasattr(mpc, 'sim_solver_full') else None
        if sim_solver is None:
            raise ValueError("No sim_solver provided, and mpc.sim_solver_full not found.")

    dt_sim = sim_solver.T  # simulation step size from the solver
    num_steps = int(duration / dt_sim)

    x_dim = 12
    u_dim = 2

    # Allocate storage for the trajectory
    x_traj = np.zeros((num_steps + 1, x_dim))
    x_traj[0] = x0
    u_traj = np.zeros((num_steps, u_dim))

    stage_costs = []
    solve_times = []
    number_of_iters = []

    # Ornstein-Uhlenbeck noise state
    noise_state = np.zeros(x_dim)

    # We'll keep a "current" control guess
    u_current = np.zeros(u_dim)

    # create the y_ref vector
    y_ref_full = np.zeros(6)
    y_ref_full[0:2] = target_xy  # set [y*, z*]

    # Main loop
    for step in range(num_steps):
        # MPC update every `control_step` steps
        if step % control_step == 0:
            # Solve the MPC with the current state
            u_current = mpc.solve(x_traj[step], target_xy)
            # For timing stats (if supported)
            solve_time = mpc.acados_ocp_solver.get_stats('time_tot')
            number_of_iters.append(mpc.acados_ocp_solver.get_stats('sqp_iter'))
            solve_times.append(solve_time)
     
        # Apply the control
        u_traj[step] = u_current

        # Compute the stage cost
        cost_val = float(stage_cost_function(x_traj[step], u_current, y_ref_full).full().item())
        stage_costs.append(cost_val)

        # Simulate one step
        sim_solver.set("x", x_traj[step])
        sim_solver.set("u", u_current)
        status = sim_solver.solve()
        if status != 0:
            print(f"Warning: sim solver returned status {status} at step {step}")

        # Update OU noise
        # dx_noise = -noise_state*(dt_sim/tau_noise) + sigma_noise*...
        dt = dt_sim
        noise_state += (-noise_state * dt / tau_noise) \
                       + (sigma_noise * np.sqrt(2*dt/tau_noise) * np.random.randn(x_dim))

        # Next state
        x_next = sim_solver.get("x") + noise_state
        x_traj[step+1] = x_next

    return x_traj, u_traj, stage_costs, solve_times, number_of_iters

