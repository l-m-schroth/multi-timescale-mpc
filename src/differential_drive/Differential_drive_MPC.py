import numpy as np
import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosMultiphaseOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
from typing import Optional, List
from dataclasses import dataclass, field
import scipy
import time
from utils_shared import get_dir

@dataclass
class DifferentialDriveMPCOptions:
    # qp_solver: str = "FULL_CONDENSING_DAQP"
    qp_solver:str = "PARTIAL_CONDENSING_HPIPM"
    sim_method_num_steps: int = 1
    cost_discretization: str = "EULER"
    globalization: str = "FIXED_STEP"
    nlp_solver_type: str = "SQP"
    # qp_solver_cond_N: int = 10
    a_max: int = 500  # Define the max force allowed
    T_max: float = 20.
    omega_max: float = .5
    N: int = 20
    integrator_type = "IRK"
    step_sizes: List[float] = field(default_factory=lambda: [0.01]*20)  
    switch_stage: int = 21
    Q_mat_full = 2 * np.diag([1e4, 1e4, 1e-4, 5e3, 1e-3, 5e-1, 5e-1])
    Q_mat_approx: np.array = 2 * np.diag([1e4, 1e4, 1e-4, 5e3, 1e-3])  # [x,y,x_d,y_d,th,th_d]
    R_mat_approx: np.array = 2 * 5 * np.diag([1e-1, 1e-1])

class DifferentialDriveMPC:
    """
    Large parts of this code are copied from https://github.com/FreyJo/ocp_solver_benchmark/blob/main/experiments/actuator_diff_drive.py and adapted for our purposes.

    This class implements the differential-drive MPC in acados. 
    Settings can be adjusted in the DifferentialDriveMPCOptions object that is passed during instantiation of DifferentialDriveMPC.
    Usage of a single- or multi-phase MPC solver can be controlled via the switch_stage.
    For switch_stage >= N: Single-phase solver that considers the actuator model over the entire horizon
    For 0 < switch_stage < N: Multi-phase solver, full model is considered until the specified stage, then the acutator dynamics are neglected.
    """

    def __init__(self, options: DifferentialDriveMPCOptions):
        self.opts = options

        # time stamp to avoid solvers interfering with each other
        self.timestamp = int(time.time()*1000)
        
        # generate ocp
        self.multi_phase = False
        if self.opts.switch_stage == 0:
            ValueError("Switch stage cannot be zero in this problem, exact model has to be used at least for a part of the horizon")
        if self.opts.switch_stage >= self.opts.N:
            self.total_ocp = self.get_diff_drive_ocp_with_actuators(options)
        else:
            # multi phase case
            self.multi_phase = True
            self.total_ocp = self.get_multiphase_ocp_actuation_to_diff_drive(self.opts)

        # Create Acados Solver
        solvers_dir = get_dir("solvers") / "differential_drive"
        solvers_dir.mkdir(parents=True, exist_ok=True)
        json_stem = f"acados_ocp_diff_drive_{self.opts.switch_stage}_{self.timestamp}"
        self.total_ocp.code_export_directory = str(solvers_dir / f"c_generated_code_{json_stem}")
        json_file = solvers_dir / f"{json_stem}.json"
        self.acados_ocp_solver = AcadosOcpSolver(self.total_ocp, json_file=str(json_file))

        # Create Acados Simulation Solver (for closed-loop simulation)
        self.acados_sim_solver, self.acados_sim_solver_no_act = self._create_sim(json_file_suffix=f"diff_drive_sim_{self.timestamp}")
        
    ### general functionality ### 
    def set_initial_state(self, x0):
        self.acados_ocp_solver.set(0, "lbx", x0)
        self.acados_ocp_solver.set(0, "ubx", x0)

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
            print(f"[DiffDriveMPC] OCP solver returned status {status}.")

        # Return first control input
        return self.acados_ocp_solver.get(0, "u")
    
    def _create_sim(self, json_file_suffix="diff_drive_sim"):
        """
        Creates an AcadosSim solver for the 3D model (useful for closed-loop simulation).
        """
        solvers_dir = get_dir("solvers") / "differential_drive"
        solvers_dir.mkdir(parents=True, exist_ok=True)

        model = self.get_unicycle_model_actuators()
        sim = AcadosSim()
        sim.model = model
        # Pick a step size for simulation (same as the first step size)
        sim.solver_options.T = self.opts.step_sizes[0]
        sim.solver_options.integrator_type = self.opts.integrator_type
        sim.code_export_directory = str(solvers_dir / f"c_generated_code_sim_{json_file_suffix}_7d")
        sim_solver_7d = AcadosSimSolver(sim, json_file=str(solvers_dir / f"acados_sim_solver_{json_file_suffix}_7d.json"))

        # Create sim with 5 dimensional model only
        model = self.get_diff_drive_model()
        sim = AcadosSim()
        sim.model = model
        # Pick a step size for simulation (same as the first step size)
        sim.solver_options.T = self.opts.step_sizes[0]
        sim.solver_options.integrator_type = self.opts.integrator_type

        sim.code_export_directory = str(solvers_dir / f"c_generated_code_sim_{json_file_suffix}_5d")
        sim_solver_5d = AcadosSimSolver(sim, json_file=str(solvers_dir / f"acados_sim_solver_{json_file_suffix}_5d.json"))
        return sim_solver_7d, sim_solver_5d

    ### Functions to generate the models """    

    def get_diff_drive_model(self) -> AcadosModel:
        """
        Creates the differential drive model without actuator model
        """
        model_name = "diff_drive"

        # set up states & controls
        x_pos = ca.SX.sym("x_pos")
        y_pos = ca.SX.sym("y_pos")
        v = ca.SX.sym("v")
        theta = ca.SX.sym("theta")
        omega = ca.SX.sym("omega")

        x = ca.vertcat(x_pos, y_pos, v, theta, omega)
        nx = x.rows()

        tau_r = ca.SX.sym("tau_r")
        tau_l = ca.SX.sym("tau_l")
        u = ca.vertcat(tau_r, tau_l)

        # xdot
        xdot = ca.SX.sym("xdot", nx)

        # dynamics
        m = 220
        mc = 200
        L = 0.32
        R = 0.16
        d = 0.01
        I = 9.6
        Iw = 0.1
        term = (tau_r + tau_l)/R
        term_2 = 2*Iw/R**2
        v_dot = (term + mc * d* omega**2)/(m + term_2)
        omega_dot = (L*(tau_r - tau_l)/R - mc*d*omega*v)/(I + L**2*term_2)
        f_expl = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), v_dot, omega, omega_dot)

        f_impl = xdot - f_expl

        model = AcadosModel()

        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = model_name

        model.t_label = "$t$ [s]"
        model.x_labels = ["$x$", "$y$", "$v$", "$\\theta$", "$\\omega$"]
        model.u_labels = [r"$\tau_r$", r"$\tau_l$"]

        return model


    def get_unicycle_model_actuators(self) -> AcadosModel:
        """
        Creates the differential drive model with actuator model
        """
        model_name = "actuators_diff_drive_with_actuators"

        # set up states & controls
        x_pos = ca.SX.sym("x_pos")
        y_pos = ca.SX.sym("y_pos")
        v = ca.SX.sym("v")
        theta = ca.SX.sym("theta")
        omega = ca.SX.sym("omega")
        x = ca.vertcat(x_pos, y_pos, v, theta, omega)

        tau_r = ca.SX.sym("tau_r")
        tau_l = ca.SX.sym("tau_l")

        # dynamics
        m = 220
        mc = 200
        L = 0.32
        R = 0.16
        d = 0.01
        I = 9.6
        Iw = 0.1
        term = (tau_r + tau_l)/R
        term_2 = 2*Iw/R**2
        v_dot = (term + mc * d* omega**2)/(m + term_2)
        omega_dot = (L*(tau_r - tau_l)/R - mc*d*omega*v)/(I + L**2*term_2)
        f_expl = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), v_dot, omega, omega_dot)

        # actuators
        #  voltage
        V_r = ca.SX.sym("V_r")
        V_l = ca.SX.sym("V_l")

        #  current
        i_r = ca.SX.sym("i_r")
        i_l = ca.SX.sym("i_l")
        K1 = 1.0  # motor constants
        K2 = 1.0
        L_inductance = 0.0001 # coil inductance
        R_resistance = 0.05 # coil resistance

        # append to ODE
        phi1_dot = (f_expl[0] * ca.cos(theta) + f_expl[1] * ca.sin(theta) + L * omega) / R
        phi2_dot = (f_expl[0] * ca.cos(theta) + f_expl[1] * ca.sin(theta) - L * omega) / R
        i_r_dot = (- K1 * phi1_dot - R_resistance * i_r + V_r) / L_inductance
        i_l_dot = (- K2 * phi2_dot - R_resistance * i_l + V_l) / L_inductance

        f_expl = ca.vertcat(f_expl, i_r_dot, i_l_dot)
        x = ca.vertcat(x, i_r, i_l)

        # substitue
        f_expl = ca.substitute(f_expl, tau_r, K1 * i_r)
        f_expl = ca.substitute(f_expl, tau_l, K2 * i_l)
        u = ca.vertcat(V_r, V_l)

        # xdot
        nx = x.rows()
        xdot = ca.SX.sym("xdot", nx)
        f_impl = xdot - f_expl

        model = AcadosModel()

        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = model_name

        model.t_label = "$t$ [s]"
        model.x_labels = ["$x$", "$y$", "$v$", "$\\theta$", "$\\omega$", "$i_r$", "$i_l$"]
        model.u_labels = [r"$V_r$", r"$V_l$"]

        return model

    def get_transition_model_actuators_to_diff_drive(self) -> AcadosModel:
        """
        Creates the transition model between the differential drive model with and without actuation
        """
        x_pos = ca.SX.sym("x_pos")
        y_pos = ca.SX.sym("y_pos")
        v = ca.SX.sym("v")
        theta = ca.SX.sym("theta")
        omega = ca.SX.sym("omega")
        i_1 = ca.SX.sym("i_1")
        i_2 = ca.SX.sym("i_2")

        x = ca.vertcat(x_pos, y_pos, v, theta, omega, i_1, i_2)

        model = AcadosModel()
        model.x = x
        model.name = "transition_actuators_to_diff_drive"
        model.disc_dyn_expr = ca.vertcat(x_pos, y_pos, v, theta, omega)
        return model
    
    ### functions to generate the ocps ###

    def get_multiphase_ocp_actuation_to_diff_drive(self,options: DifferentialDriveMPCOptions) -> AcadosOcp:
        N_horizon_0 = options.switch_stage
        N_horizon_1 = options.N - options.switch_stage

        ocp = AcadosMultiphaseOcp([N_horizon_0, 1, N_horizon_1])
        ocp_0 = self.get_diff_drive_ocp_with_actuators(options)
        ocp.set_phase(ocp_0, 0)

        # transition ocp.
        transition_ocp = AcadosOcp()
        model = self.get_transition_model_actuators_to_diff_drive()
        transition_ocp.model = model
        transition_ocp.cost.cost_type = "NONLINEAR_LS"
        transition_ocp.model.cost_y_expr = model.x
        transition_ocp.cost.W = 1e-7 * np.eye(model.x.rows())
        transition_ocp.cost.W[-2:, -2:] = 0.1*options.step_sizes[self.opts.switch_stage-1] * 1e0 * np.eye(2)
        transition_ocp.cost.yref = np.zeros((model.x.rows(),))
        ocp.set_phase(transition_ocp, 1)

        ocp_1 = self.get_diff_drive_ocp_no_actuators(options)
        ocp.set_phase(ocp_1, 2)
        ocp.solver_options = ocp_0.solver_options
        ocp.solver_options.tf = sum(options.step_sizes) + 1
        step_sizes_list_with_transition = list(self.opts.step_sizes[:self.opts.switch_stage]) + [1.0] + list(self.opts.step_sizes[self.opts.switch_stage:])
        ocp.solver_options.time_steps = np.array(step_sizes_list_with_transition)

        ocp.mocp_opts.cost_discretization = [options.cost_discretization, "EULER", options.cost_discretization]
        ocp.mocp_opts.integrator_type = [options.integrator_type, "DISCRETE", "ERK"]
        return ocp


    def set_solver_options_in_ocp(self, ocp: AcadosOcp, options: DifferentialDriveMPCOptions):
        # set options
        ocp.solver_options.qp_solver = options.qp_solver
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = options.integrator_type
        ocp.solver_options.nlp_solver_type = options.nlp_solver_type
        ocp.solver_options.sim_method_num_stages = 3
        ocp.solver_options.sim_method_num_steps = options.sim_method_num_steps
        ocp.solver_options.globalization = options.globalization
        ocp.solver_options.cost_discretization = options.cost_discretization
        ocp.solver_options.qp_solver_iter_max = 1000
        ocp.solver_options.nlp_solver_max_iter = 600
        ocp.solver_options.levenberg_marquardt = 1e-4
        # ocp.solver_options.qp_solver_cond_N = options.qp_solver_cond_N
        ocp.solver_options.tol = 1e-6
        ocp.solver_options.qp_tol = 1e-2 * ocp.solver_options.tol
        # ocp.solver_options.print_level = 2
        # ocp.solver_options.nlp_solver_ext_qp_res = 1

    def get_diff_drive_ocp_no_actuators(self, options: DifferentialDriveMPCOptions) -> AcadosOcp:
        N_horizon = options.N

        # create ocp object to formulate the OCP
        ocp = AcadosOcp()
        model = self.get_diff_drive_model()
        ocp.model = model
        nx = model.x.rows()
        nu = model.u.rows()

        # set dimensions
        ocp.solver_options.N_horizon = N_horizon
        #ocp.solver_options.tf = options.T_horizon

        # set cost
        Q_mat = self.opts.Q_mat_approx
        R_mat = self.opts.R_mat_approx

        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        ny = nx + nu
        ny_e = nx

        ocp.cost.W_e = Q_mat
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)

        Vu = np.zeros((ny, nu))
        Vu[nx : nx + nu, 0:nu] = np.eye(nu)
        ocp.cost.Vu = Vu

        ocp.cost.Vx_e = np.eye(nx)

        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((ny_e,))
     
        # set constraints
        ocp.constraints.idxbu = np.arange(nu)
        ocp.constraints.ubu = np.array([60, 60])
        ocp.constraints.lbu = -ocp.constraints.ubu
        ocp.constraints.idxbx = np.array([2, 4])
        ocp.constraints.ubx = np.array([1., self.opts.omega_max])
        ocp.constraints.lbx = np.array([0., -self.opts.omega_max])

        ocp.constraints.x0 = np.zeros(nx)

        self.set_solver_options_in_ocp(ocp, options)
        return ocp

    def get_diff_drive_ocp_with_actuators(self, options: DifferentialDriveMPCOptions) -> AcadosOcp:
        N_horizon = options.N

        # create ocp object to formulate the OCP
        ocp = AcadosOcp()
        model = self.get_unicycle_model_actuators()
        ocp.model = model
        nx = model.x.rows()
        nu = model.u.rows()

        # set dimensions
        ocp.solver_options.N_horizon = N_horizon
        ocp.solver_options.tf = sum(options.step_sizes)
        ocp.solver_options.time_steps = np.array(options.step_sizes)

        # set cost
        Q_mat = self.opts.Q_mat_full # [x_pos, y_pos, v, theta, omega, i_1, i_2]
        # R_mat = 2 * 5 * np.diag([1e-5, 1e-5])

        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        ny = nx
        ny_e = nx

        ocp.cost.W_e = Q_mat
        ocp.cost.W = Q_mat

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)

        Vu = np.zeros((ny, nu))
        # Vu[nx : nx + nu, 0:nu] = np.eye(nu)
        ocp.cost.Vu = Vu

        ocp.cost.Vx_e = np.eye(nx)

        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((ny_e,))

        # set constraints
        ocp.constraints.idxbu = np.arange(nu)
        ocp.constraints.ubu = np.array([10.0, 10.0])
        ocp.constraints.lbu = -ocp.constraints.ubu

        # bound on v, omega
        ocp.constraints.idxbx = np.array([2, 4])
        ocp.constraints.ubx = np.array([1., self.opts.omega_max])
        ocp.constraints.lbx = np.array([0., -self.opts.omega_max])

        ocp.constraints.x0 = np.zeros(nx)

        # set options
        self.set_solver_options_in_ocp(ocp, options)

        return ocp
    


   
        
