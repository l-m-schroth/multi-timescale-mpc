from dataclasses import dataclass
import time
from acados_template import AcadosOcpSolver, AcadosSim, AcadosSimSolver, AcadosOcp, AcadosMultiphaseOcp
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import block_diag
from utils_shared import compute_exponential_step_sizes
from utils_shared import get_dir
from copy import deepcopy
import gc
import shutil
import os
import pickle
import matplotlib.pyplot as plt
import casadi as ca

@dataclass 
class FFMPCOptions(ABC):
    # contains all options
    N: int = 20 # number of shooting stages of the FFMPC
    T: float = 1.0 # time horizon of the FFMPC
    dt_initial = 0.05 # The exponential step size schedule will be computed based on N, dt_initial and T
    switch_stage_k_bar: int = 20
    integrator_type_full_model: str = "IRK"
    integrator_type_reduced_model: str = "ERK"
    # cost matrices for first phase
    Q_full: np.ndarray | None = None
    R_full: np.ndarray | None = None
    Q_red: np.ndarray | None = None
    R_red: np.ndarray | None = None
    system_name: str = "my_system" # a string containing your system's name
    plot_step_size_schedule: bool = False # whether to plot to generated exponential step size schedule
    # decides whether to attach the time stamp to the .json file name. 
    # This leads to recreation of solvers when a new FFMPC Object is created.
    # Currently the safer option, as we observed issues with solvers interferring with each other otherwhise.
    include_timestamp_in_json_names: bool = True
    # This needs to be activated if non-zero reference values are used with nonlinear LS costs
    overwrite_y_ref: bool = False

class FFMPC:
    
    """
    Add docu
    """

    def __init__(self, options: FFMPCOptions):
        self.opts = options

        # time stamp to avoid solvers interfering with each other
        self.timestamp = int(time.time()*1000)

        # determine exponential step size schedule based on options
        self.step_sizes = compute_exponential_step_sizes(self.opts.dt_initial, self.opts.T, self.opts.N, plot=self.opts.plot_step_size_schedule)
        
        # generate ocp
        self.multi_phase = False
        if self.opts.switch_stage_k_bar == 0:
            raise ValueError("Switch stage cannot be zero, exact model has to be used at least for a part of the horizon in ffmpc")
        if self.opts.switch_stage_k_bar >= self.opts.N:
            model = self.get_full_model()
            # save dimensions
            self.nx_full = model.x.rows()
            self.nu_full = model.u.rows()
            cost_y_expr, cost_y_expr_e = self.cost_expression_full(model) 
            total_ocp = self._get_ocp(model, self.opts.Q_full, self.opts.R_full, cost_y_expr, cost_y_expr_e)
            self.total_ocp = self.set_constraints_in_ocp_full(total_ocp) # overwrites constraints
            # initialize stage cost function for evaluation
            self._init_stage_cost_function_full(model, cost_y_expr)
        else:
            # multi phase case
            self.multi_phase = True
            self.total_ocp = self._get_multiphase_ocp()

        # Create Acados Solver
        if self.opts.include_timestamp_in_json_names:
            json_stem = f"ffmpc_acados_ocp_{self.opts.system_name}_{self.opts.switch_stage_k_bar}_{self.timestamp}"
        else:
            json_stem = f"ffmpc_acados_ocp_{self.opts.system_name}_{self.opts.switch_stage_k_bar}"

        solvers_dir = get_dir("solvers") / "ffmpc"
        solvers_dir.mkdir(parents=True, exist_ok=True)
        self.total_ocp.code_export_directory = str(solvers_dir / f"c_generated_code_{json_stem}")
        json_file = solvers_dir / f"{json_stem}.json"
        self.acados_ocp_solver = AcadosOcpSolver(self.total_ocp, json_file=str(json_file))

        # Acados simulation solver dict to save sim solvers of different dt 
        self.acados_sim_solvers = {}

    # --- functions that need to be overwritten by the user --- #
    
    @abstractmethod
    def get_full_model(self):
        pass

    @abstractmethod
    def get_reduced_model(self):
        pass

    @abstractmethod
    def get_transition_model(self):
        pass
    
    @abstractmethod
    def cost_expression_full(self, model):
        """casadi cost expression for full model phase"""
        pass
    
    @abstractmethod
    def cost_expression_reduced(self, model):
        """casadi cost expression for reduced model phase"""
        pass

    # this only needs to be overwritten if the user wants to change the default solver options
    def overwrite_default_solver_options(self, ocp):
        return ocp

    # this only needs to be overwritten if the user wants to add constraints
    def set_constraints_in_ocp_full(self, ocp):
        return ocp

    def set_constraints_in_ocp_reduced(self, ocp):
        return ocp
    
    def set_transition_costs(self, transition_ocp, model_trans):
        return transition_ocp
    
    # this only needs to be overwritten if user wants a closed-loop trajectory plot.
    def plot_closed_loop_trajectory(self, x_traj, u_traj):
        pass

    # this only needs to be overwritten to set non-zero target values, for instance for trajectory tracking
    def get_y_ref(self, t, terminal=False, phase="full"):
        """if user needs different behaviour for full model and reduced model phase, use if_else for "full" and "reduced" phase"""
        pass

    # --- general functionality --- # 

    def set_initial_state(self, x0):
        self.acados_ocp_solver.set(0, "lbx", x0)
        self.acados_ocp_solver.set(0, "ubx", x0)

    # useful for retrieving open loop plans, currently not in use in this class
    def get_planned_trajectory(self):
        N = self.opts.N
        if self.multi_phase:
            N += 1 # in multi-phase case, we need to account for the switching stage
        traj_x, traj_u = [], []
        for i in range(N):  # N+1 due to transition stage
            x_i = self.acados_ocp_solver.get(i, "x")
            u_i = self.acados_ocp_solver.get(i, "u")
            traj_x.append(x_i)
            traj_u.append(u_i)
        x_N = self.acados_ocp_solver.get(N, "x")
        traj_x.append(x_N)
        return traj_x, traj_u

    def solve(self, x0):
        """
        Solves the MPC problem:
        1. Sets the initial guess with x0.
        3. Solves the OCP.
        """
        # Set initial guess
        self.set_initial_state(x0)

        # Solve the OCP
        status = self.acados_ocp_solver.solve()
        
        if status != 0:
            print(f"[DiffDriveMPC] OCP solver failed or did not converge, returned status {status}.")

        # Return first control input
        return self.acados_ocp_solver.get(0, "u")
    
    # --- sim solver creation --- # 

    def _create_sim_solver(self, dt, json_file_suffix="sim_solver"):
        """
        Creates an AcadosSim solver for the full model (useful for closed-loop simulation).
        """
        model = self.get_full_model()
        sim = AcadosSim()
        sim.model = model

        # Pick a step size for simulation (same as the first step size)
        sim.solver_options.T = dt 
        sim.solver_options.integrator_type = "IRK" # integration with more accurate IRK for simulation

        solvers_dir = get_dir("solvers") / "ffmpc"
        solvers_dir.mkdir(parents=True, exist_ok=True)
        json_stem = f"acados_sim_solver_{json_file_suffix}_{dt}"
        sim.code_export_directory = str(solvers_dir / f"c_generated_code_{json_stem}")
        sim_solver = AcadosSimSolver(sim, json_file=str(solvers_dir / f"{json_stem}.json"))
        self.acados_sim_solvers[f"{dt}"] = sim_solver

        return sim_solver
    
    # --- functions to generate the ocps --- # 

    def set_additional_solver_options_in_ocp(self, ocp: AcadosOcp):
        #  additional solver options to set besides horizon, tf and time_steps
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = self.opts.integrator_type_full_model
        ocp.solver_options.nlp_solver_type = "SQP"
        return ocp

    def _get_ocp(self, model, Q, R, cost_y_expr, cost_y_expr_e):

        # create ocp object and retrieve model
        ocp = AcadosOcp()
        ocp.model = model
        nx = model.x.rows()
        nu = model.u.rows()

        # solver options for ocp
        ocp.solver_options.N_horizon = self.opts.N
        ocp.solver_options.tf = sum(self.step_sizes)
        ocp.solver_options.time_steps = np.array(self.step_sizes)
        ocp = self.set_additional_solver_options_in_ocp(ocp)
        ocp = self.overwrite_default_solver_options(ocp)

        # create tracking costs
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        ocp.cost.W = block_diag(Q, R)
        ocp.cost.W_e = Q

        ocp.model.cost_y_expr, ocp.model.cost_y_expr_e = cost_y_expr, cost_y_expr_e
        ny = cost_y_expr.shape[0]
        ny_e = cost_y_expr_e.shape[0] 

        # dummy reference set top zero, if want to control to another state or do trajectory tracking, we must overwrite
        ocp.cost.yref = np.zeros((ny,)) 
        ocp.cost.yref_e = np.zeros((ny_e,))  

        # constraints
        ocp.constraints.x0 = np.zeros(nx)

        return ocp

    def _get_multiphase_ocp(self):
        self.N_full = self.opts.switch_stage_k_bar
        self.N_reduced = self.opts.N - self.opts.switch_stage_k_bar
        ocp = AcadosMultiphaseOcp([self.N_full, 1, self.N_reduced])
        
        # ocp for full model phase 
        model_full = self.get_full_model()
        self.nx_full = model_full.x.rows()
        self.nu_full = model_full.u.rows()
        cost_y_expr_full, cost_y_expr_e_full = self.cost_expression_full(model_full)
        ocp_full = self._get_ocp(model_full, self.opts.Q_full, self.opts.R_full, cost_y_expr_full, cost_y_expr_e_full)
        ocp_full = self.set_constraints_in_ocp_full(ocp_full) # overwrites constraints
        ocp.set_phase(ocp_full, 0)
        # initialize stage cost function for evaluation
        self._init_stage_cost_function_full(model_full, cost_y_expr_full)

        # transition ocp 
        transition_ocp = AcadosOcp()
        model_trans = self.get_transition_model()
        transition_ocp.model = model_trans
        # default is no transition cost
        transition_ocp.cost.cost_type = "NONLINEAR_LS"
        transition_ocp.model.cost_y_expr = model_trans.x
        transition_ocp.cost.W = 0.0 * np.eye(model_trans.x.rows())
        transition_ocp.cost.yref = np.zeros((model_trans.x.rows(),))
        transition_ocp = self.set_transition_costs(transition_ocp, model_trans)
        ocp.set_phase(transition_ocp, 1)
        
        # ocp for reduced model phase
        model_red = self.get_reduced_model()
        cost_y_expr_red, cost_y_expr_e_red = self.cost_expression_reduced(model_red)
        ocp_red = self._get_ocp(model_red, self.opts.Q_red, self.opts.R_red, cost_y_expr_red, cost_y_expr_e_red)
        ocp_red = self.set_constraints_in_ocp_reduced(ocp_red) # overwrites constraints
        ocp.set_phase(ocp_red, 2)

        # settings for multi-phase ocp, overwrites these settings specified for the individual ocps
        ocp.solver_options = ocp_full.solver_options # most solver options as in full model phase
        ocp.solver_options.tf = sum(self.step_sizes) + 1
        step_sizes_list_with_transition = list(self.step_sizes[:self.opts.switch_stage_k_bar]) + [1.0] + list(self.step_sizes[self.opts.switch_stage_k_bar:])
        ocp.solver_options.time_steps = np.array(step_sizes_list_with_transition)
        ocp = self.set_additional_solver_options_in_ocp(ocp)

        ocp.mocp_opts.integrator_type = [self.opts.integrator_type_full_model, "DISCRETE", self.opts.integrator_type_reduced_model]

        return ocp

    # --- closed loop simulation functionality --- # 
    def simulate_closed_loop(self, x0, dt_sim, sim_duration, plot=False):

        # check if sim_solver with the desired dt is already generated, if not -> generate
        if f"{dt_sim}" in self.acados_sim_solvers:
            sim_solver = self.acados_sim_solvers[f"{dt_sim}"]
        else:
            sim_solver = self._create_sim_solver(dt_sim)

        # determine control decimation
        if self.opts.dt_initial % dt_sim != 0:
            print(f"Warning: dt_initial ({self.opts.dt_initial}) "
                f"is not a multiple of dt_sim ({dt_sim})")
        control_decimation = int(self.opts.dt_initial / dt_sim)

        # determine number of closed loop steps
        num_steps = int(sim_duration / dt_sim)
        t_sim = 0

        # Allocate storage for the trajectory
        x_traj = np.zeros((num_steps + 1, self.nx_full))
        x_traj[0] = x0
        u_traj = np.zeros((num_steps, self.nu_full))

        # statistics
        stage_costs = []
        solve_times = []

        for step in range(num_steps):
            # control inputs updated taking control decimation into account
            if step % control_decimation == 0:
                # If non-trivial y_ref values are used, set them accordingly before calling the solve function
                if self.opts.overwrite_y_ref:
                    self.set_reference_trajectory(t_sim)
                # Solve the MPC with the current state
                u = self.solve(x_traj[step])
                # For timing stats (if supported)
                solve_times.append(self.acados_ocp_solver.get_stats('time_tot'))

            # Apply the control
            u_traj[step] = u

            # compute the stage-costs
            yref_now = self.get_y_ref(t_sim, phase="full")
            stage_cost_val = float(np.array(self.stage_costs_fct_full(x_traj[step], u, yref_now)).squeeze())
            stage_costs.append(stage_cost_val)

            # Simulate one step
            sim_solver.set("x", x_traj[step])
            sim_solver.set("u", u)
            status = sim_solver.solve()
            if status != 0:
                print(f"Warning: sim solver returned unusual status {status} at step {step}")
            
            # Next state
            x_next = sim_solver.get("x")
            x_traj[step+1] = x_next

            # advance simulation time
            t_sim += dt_sim

        if plot:
            self.plot_closed_loop_trajectory(x_traj, u_traj)

        return x_traj, u_traj, stage_costs, solve_times
    
    def _init_stage_cost_function_full(self, model, cost_y_expr):
        """
        Create a CasADi Function ℓ(x,u,yref) = (y(x,u) - yref)^T W (y(x,u) - yref)
        for the FULL model phase, where y = cost_y_expr and W = blkdiag(Q_full, R_full).
        """
        # W must match the dimension of cost_y_expr
        W_full_np = block_diag(self.opts.Q_full, self.opts.R_full)
        W_full = ca.DM(W_full_np)                   

        ny = int(cost_y_expr.shape[0])
        yref_sym = ca.MX.sym("yref_full", ny)        

        res = cost_y_expr - yref_sym
        l_stage = ca.mtimes([res.T, W_full, res])   

        # Expose as f(x,u,yref)->scalar
        self.stage_costs_fct_full = ca.Function(
            "stage_costs_fct_full",
            [model.x, model.u, yref_sym],
            [l_stage],
            ["x", "u", "yref"],
            ["ell"]
        )

        
    # --- functionality for sweep over switching times --- # 
    def switching_stage_sweep(self, x0, dt_sim, sim_duration, build_fn=None):

        switching_stages = list(range (1, self.opts.N))
        mean_costs_sweep = []

        ffmpc_options_sweep = deepcopy(self.opts)
        for switching_stage in switching_stages:
            ffmpc_options_sweep.switch_stage_k_bar = switching_stage
            ffmpc_sweep = build_fn(ffmpc_options_sweep) # build function for subclass used here
            _, _, stage_costs, _ = ffmpc_sweep.simulate_closed_loop(x0, dt_sim, sim_duration)

            # clean up 
            del ffmpc_sweep
            gc.collect()
            shutil.rmtree('c_generated_code', ignore_errors=True)

            # mean values
            mean_costs_sweep.append(np.mean(stage_costs))

        # evaluate full-resolution baseline with same time horion, but full model and small time steps
        ffmpc_options_baseline = deepcopy(self.opts)
        ffmpc_options_baseline.N = int(self.opts.T / self.opts.dt_initial) # no step size increase
        ffmpc_options_baseline.switch_stage_k_bar = ffmpc_options_baseline.N + 1 # no model switching
        mpc_baseline = build_fn(ffmpc_options_baseline) 
        _, _, stage_costs_baseline, _ = mpc_baseline.simulate_closed_loop(x0, dt_sim, sim_duration)
        mean_costs_baseline = np.mean(stage_costs_baseline)

        # path to directory in which sweep results are saved
        data_dir = get_dir("data")
        results_file_sweep = data_dir / f"{self.opts.system_name}/switching_index_sweep.pkl" 

        # dump the results in the data directory
        data = {
            'mean_costs_sweep': mean_costs_sweep,
            'switching_stages': switching_stages,
            'mean_costs_baseline': mean_costs_baseline
        }
        os.makedirs(os.path.dirname(results_file_sweep), exist_ok=True)
        with open(results_file_sweep, 'wb') as f:
            pickle.dump(data, f)

        # plots the results and save the plot in the plots directory
        self.plot_switching_stage_sweep_results(results_file_sweep)

    def plot_switching_stage_sweep_results(self, results_file_sweep):

        # load results via pickle
        if os.path.exists(results_file_sweep):
            with open(results_file_sweep, 'rb') as f:
                data = pickle.load(f)
            mean_costs_sweep = data['mean_costs_sweep']
            switching_stages = data['switching_stages']
            mean_costs_baseline = data['mean_costs_baseline']
        else:
            raise FileNotFoundError(f"Data file not found, please check path: '{results_file_sweep}'")
        
        plt.figure(figsize=(8,5))
        plt.plot(switching_stages[1:], 100*np.array(mean_costs_sweep[1:] - mean_costs_baseline)/ mean_costs_baseline, marker='o', linestyle='-', linewidth=2)

        plt.xlabel(r"Switching stage $\bar{k}$", fontsize=12)
        plt.ylabel("Mean closed-loop cost increase [%]", fontsize=12)
        plt.yscale('log')  
        #plt.title("Mean Closed Loop Costs vs. Switching Index", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        # save plot in plots directory
        data_dir = get_dir("plots")
        plt.savefig(data_dir / f"{self.opts.system_name}_costs_vs_switching_index.pdf", bbox_inches="tight")

        # show plot
        plt.show()

    # --- useful functionality for trajectory tracking tasks --- # 
    def set_reference_trajectory(self, t0):
        if self.multi_phase:
            # Multi-phase case
            index_offset = 0
            t_eval = t0
            t_update_counter = 0
            phase = "full"
            for phase, N_phase in enumerate([self.N_full, self.N_reduced]):
                if index_offset > 0:
                    phase = "reduced"
                for i in range(N_phase):
                    y_ref = self.get_y_ref(t_eval, phase=phase)
                    self.acados_ocp_solver.set(index_offset + i, "yref", y_ref)
                    t_eval += self.step_sizes[t_update_counter]
                    t_update_counter += 1

                # Update offset and move to the next phase
                index_offset += N_phase + 1  # Skip transition stage

            # Set terminal reference for the phase
            last_idx = index_offset - 1
            y_ref_N = self.get_y_ref(t_eval, terminal=True, phase=phase)
            self.acados_ocp_solver.set(last_idx, "yref", y_ref_N)   
        else:
            # Single-phase case
            N = self.opts.N
            t_eval = t0
            for i in range(N):
                y_ref = self.get_y_ref(t_eval, phase="full")
                self.acados_ocp_solver.set(i, "yref", y_ref)
                t_eval += self.step_sizes[i]
            
            # Set terminal reference
            y_ref_N = self.get_y_ref(t_eval, terminal=True, phase="full")
            self.acados_ocp_solver.set(N, "yref", y_ref_N)
            

        

        


