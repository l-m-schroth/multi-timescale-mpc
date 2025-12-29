import numpy as np
import casadi as ca
import time
from dataclasses import dataclass, field
from typing import Optional, List
from utils_shared import get_dir

from acados_template import (
    AcadosModel,
    AcadosOcp,
    AcadosMultiphaseOcp,
    AcadosOcpSolver,
    AcadosSim,
    AcadosSimSolver,
)

###############################################################################
# Options Dataclass
###############################################################################

@dataclass
class DroneMPCOptions:
    """
    Holds relevant parameters for configuring the DroneMPC.
    """
    # Horizon & discretization
    N: int = 20
    switch_stage: int = 20
    step_sizes: List[float] = field(default_factory=lambda: [0.01]*20)

    # Solver / integrator settings
    nlp_solver_type: str = "SQP"
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM" # "PARTIAL_CONDENSING_OSQP" 
    integrator_type: str = "IRK"
    levenberg_marquardt: str = 8e-3

    # Physical parameters
    M: float = 2.0       # drone mass in kg
    m: float = 0.1       # load mass in kg (small compared to drone mass)
    Ixx: float = 0.05    # moment of inertia (kg·m²)
    g: float = 9.81      # gravitational acceleration (m/s²)
    c: float = 1.0       # rotor thrust constant (N per unit input)
    L_rot: float = 0.2   # half distance between rotors in m

    # Pendulum parameters (used only if pendulum is active)
    k: float = 100.0     # spring stiffness (N/m)
    l0: float = 0.2      # spring rest length in m

    # Cost weighting matrices
    Q: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0]))        # on [y, z]
    Q_acc_load: np.ndarray = field(default_factory=lambda: np.diag([0.1, 0.1])) # on [y_load_dd, z_load_dd]
    R_reg: np.ndarray = field(default_factory=lambda: np.diag([1e-3, 1e-3])) # very small penalty on inputs for regularization
    Q_trans: np.ndarray = field(default_factory=lambda: np.diag([0.00, 0.00])) # no transition costs

    # Constraints
    w_min: float = -10.0
    w_max: float = 10.0
    w_dot_min: float = -5.0
    w_dot_max: float = 5.0
    phi_min: float = -0.7
    phi_max: float = 0.7
    F_min: float = -15.0
    F_max: float = 15.0

    # Whether to create a sim solver for the full model
    create_sim: bool = True


###############################################################################
# Main Drone MPC Class
###############################################################################

class DroneMPC:
    """
    This class implements the drone MPC in acados. 
    Settings can be adjusted in the DroneMPCOptions object that is passed during instantiation of DroneMPC.
    Usage of a single- or multi-phase MPC solver can be controlled via the switch_stage.
    For switch_stage >= N: Single-phase solver that considers the actuator model over the entire horizon
    For 0 < switch_stage < N: Multi-phase solver, full model is considered until the specified stage, then the spring neglected and a stiff pendulum.
    """

    def __init__(self, options: DroneMPCOptions):
        self.opts = options
        self.timestamp = int(time.time() * 1000)

        # Flag to indicate multi-phase usage
        self.multi_phase = False

        # Build the appropriate OCP based on switch_stage
        if self.opts.switch_stage == 0:
            # Single-phase, ignoring pendulum (but with actuator model).
            self._create_single_phase_ocp_rigid_pendulum()
        elif self.opts.switch_stage >= self.opts.N:
            # Single-phase, full pendulum + actuator
            self._create_single_phase_ocp_full_pendulum()
        else:
            # Multi-phase OCP
            self.multi_phase = True
            self._create_multi_phase_ocp()

        # Create the Acados solver
        solvers_dir = get_dir("solvers") / "drone"
        solvers_dir.mkdir(parents=True, exist_ok=True)
        json_stem = f"acados_ocp_drone_{self.opts.switch_stage}_{self.timestamp}"
        self.total_ocp.code_export_directory = str(solvers_dir / f"c_generated_code_{json_stem}")
        json_file = solvers_dir / f"{json_stem}.json"
        self.acados_ocp_solver = AcadosOcpSolver(self.total_ocp, json_file=str(json_file))

        # Optionally create a single AcadosSimSolver for the full pendulum model only
        if self.opts.create_sim:
            self._create_sim_full_model()


    ###########################################################################
    # Public Methods
    ###########################################################################

    def solve(self, x0: np.ndarray, y_ref: Optional[np.ndarray] = None):
        """
        - Set initial state for the first phase.
        - If a reference is given, set it for all stages in the cost.
        - Solve the OCP and return the first control input.
        """

        if self.opts.switch_stage == 0:
            x0= self.to_10d(x0)

        # Phase 0 (index 0) gets x0 constraints
        self.acados_ocp_solver.set(0, "lbx", x0)
        self.acados_ocp_solver.set(0, "ubx", x0)

        # If reference is provided, set it (assuming y_ref = [y*, z*])
        if y_ref is not None:
            self._set_reference_in_cost(y_ref)

        status = self.acados_ocp_solver.solve()
        if status != 0:
            print(f"[DroneMPC] solver returned status {status}.")

        # Return the first control input
        return self.acados_ocp_solver.get(0, "u")

    def get_planned_trajectory(self):
        """
        Returns (traj_x, traj_u) for the entire horizon.
        If multi-phase, note there's an extra transition stage => total = N + 1.
        """
        N_total = self.opts.N
        if self.multi_phase:
            N_total += 1

        traj_x = []
        traj_u = []
        for i in range(N_total):
            x_i = self.acados_ocp_solver.get(i, "x")
            u_i = self.acados_ocp_solver.get(i, "u")
            traj_x.append(x_i)
            traj_u.append(u_i)

        x_end = self.acados_ocp_solver.get(N_total, "x")
        traj_x.append(x_end)
        return traj_x, traj_u
    
    def set_initial_guess(self, x0_full: np.ndarray, u_guess: np.ndarray):
        # Single-phase cases:
        if not self.multi_phase:
            # We have a horizon of N steps => N shooting intervals, so N+1 states
            N_horizon = self.opts.N
            if self.opts.switch_stage == 0:
                x0 = self.to_10d(x0_full)
            else:
                x0 = x0_full   
            for stage in range(N_horizon):
                self.acados_ocp_solver.set(stage, "x", x0)
                self.acados_ocp_solver.set(stage, "u", u_guess)
            self.acados_ocp_solver.set(N_horizon, "x", x0)
        else:
            N0 = self.opts.switch_stage       # horizon for phase 0
            N2 = self.opts.N - N0             # horizon for phase 2
            for stage in range(N0):
                self.acados_ocp_solver.set(stage, "x", x0_full)
                self.acados_ocp_solver.set(stage, "u", u_guess)

            # The discrete transition at stage N0: no input
            self.acados_ocp_solver.set(N0, "x", x0_full)

            x0_simpl = self.to_10d(x0_full)  # or to_10d if needed
            for stage in range(N0 + 1, N0 + 1 + N2):
                self.acados_ocp_solver.set(stage, "x", x0_simpl)
                self.acados_ocp_solver.set(stage, "u", u_guess)
            # terminal node
            self.acados_ocp_solver.set(N0 + 1 + N2, "x", x0_simpl)

    def to_10d(self, x_full):
        """Rigid pendulum + actuator => 10D."""
        return np.array([
            x_full[0],   # y
            x_full[1],   # z
            x_full[2],   # phi
            x_full[4],   # thetha
            x_full[5],   # y_dot
            x_full[6],   # z_dot
            x_full[7],   # phi_dot
            x_full[9],   # theta_dot
            x_full[10],  # w1
            x_full[11],  # w2
        ])

    ###########################################################################
    # Single-Phase OCPs
    ###########################################################################

    def _create_single_phase_ocp_rigid_pendulum(self):
        """
        Single-phase with rigid pendulum.
        """
        ocp = AcadosOcp()

        # Effective mass includes the small load mass
        M_eff = self.opts.M + self.opts.m
        model = self._build_drone_rigid_pendulum_actuator_model(M_eff)
        ocp.model = model

        ocp.solver_options.N_horizon = self.opts.N
        ocp.solver_options.tf = sum(self.opts.step_sizes)

        # Nonlinear LS cost:
        cost_expr = self._cost_expr_drone_rigid_pendulum(model)  
        ocp.model.cost_y_expr = cost_expr
        ocp.cost.cost_type = "NONLINEAR_LS"

        # Terminal cost on [y, z] => first 2 components
        ocp.model.cost_y_expr_e = ca.vertcat(cost_expr[0], cost_expr[1])
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        # Stage weighting
        W_stage = np.block([
            [self.opts.Q,             np.zeros((2, 2)),         np.zeros((2, 2))],
            [np.zeros((2, 2)),        self.opts.Q_acc_load,          np.zeros((2, 2))],
            [np.zeros((2, 2)),        np.zeros((2, 2)),          self.opts.R_reg]
        ])
        ocp.cost.W = W_stage
        ocp.cost.W_e = self.opts.Q

        ocp.cost.yref = np.zeros(6)   # [y, z, y_dd, z_dd]
        ocp.cost.yref_e = np.zeros(2) # [y, z]

        # Constraints
        nx = model.x.size()[0]
        ocp.constraints.x0 = np.zeros(nx)

        # phi => index 2
        ocp.constraints.idxbx = np.array([2])
        ocp.constraints.ubx = np.array([self.opts.phi_max])
        ocp.constraints.lbx = np.array([self.opts.phi_min])

        # w1,w2 => indices 8,9
        ocp.constraints.idxbx = np.append(ocp.constraints.idxbx, [8, 9])
        ocp.constraints.ubx = np.append(ocp.constraints.ubx, [self.opts.w_max, self.opts.w_max])
        ocp.constraints.lbx = np.append(ocp.constraints.lbx, [self.opts.w_min, self.opts.w_min])

        # controls => [dw1, dw2]
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.ubu = np.array([self.opts.w_dot_max, self.opts.w_dot_max])
        ocp.constraints.lbu = np.array([self.opts.w_dot_min, self.opts.w_dot_min])

        self._set_ocp_solver_options(ocp)
        self.total_ocp = ocp


    def _create_single_phase_ocp_full_pendulum(self):
        """
        Single-phase with spring-pendulum
        """
        ocp = AcadosOcp()

        # Build the full pendulum model from external ODE file
        model_full = self._build_full_pendulum_model()
        ocp.model = model_full

        ocp.solver_options.N_horizon = self.opts.N
        ocp.solver_options.time_steps = np.array(self.opts.step_sizes)
        ocp.solver_options.tf = sum(self.opts.step_sizes)

        # Nonlinear LS cost
        cost_expr = self._cost_expr_full_pendulum(model_full)  # dimension 4
        ocp.model.cost_y_expr = cost_expr
        ocp.cost.cost_type = "NONLINEAR_LS"

        # Terminal cost on first 2 comps => [y, z]
        ocp.model.cost_y_expr_e = ca.vertcat(cost_expr[0], cost_expr[1])
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        W_stage = np.block([
            [self.opts.Q,             np.zeros((2, 2)),         np.zeros((2, 2))],
            [np.zeros((2, 2)),        self.opts.Q_acc_load,     np.zeros((2, 2))],
            [np.zeros((2, 2)),        np.zeros((2, 2)),         self.opts.R_reg]
        ])
        ocp.cost.W = W_stage
        ocp.cost.W_e = self.opts.Q

        ocp.cost.yref = np.zeros(6)
        ocp.cost.yref_e = np.zeros(2)

        # store stage cost function with full model, this is used for evaluating the costs in closed loop simulation
        yref_ca = ca.MX.sym("yref", 6)

        diff = cost_expr - yref_ca
        stage_cost_sym = diff.T @ W_stage @ diff

        # 4) Create a CasADi function
        self.stage_cost_func_full = ca.Function(
            "stage_cost_func_full",
            [ocp.model.x, ocp.model.u, yref_ca],
            [stage_cost_sym],
            ["x", "u", "y_ref"],
            ["stage_cost"]
        )

        # Constraints
        nx = model_full.x.size()[0]
        ocp.constraints.x0 = np.zeros(nx)

        # phi => index 2
        ocp.constraints.idxbx = np.array([2])
        ocp.constraints.ubx = np.array([self.opts.phi_max])
        ocp.constraints.lbx = np.array([self.opts.phi_min])

        # w1,w2 => indices 10,11
        ocp.constraints.idxbx = np.append(ocp.constraints.idxbx, [10, 11])
        ocp.constraints.ubx = np.append(ocp.constraints.ubx, [self.opts.w_max, self.opts.w_max])
        ocp.constraints.lbx = np.append(ocp.constraints.lbx, [self.opts.w_min, self.opts.w_min])

        # control => [dw1, dw2]
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.ubu = np.array([self.opts.w_dot_max, self.opts.w_dot_max])
        ocp.constraints.lbu = np.array([self.opts.w_dot_min, self.opts.w_dot_min])

        self._set_ocp_solver_options(ocp)
        self.total_ocp = ocp


    ###########################################################################
    # Multi-Phase OCP
    ###########################################################################

    def _create_multi_phase_ocp(self):
        """
        Phase 0: Sring-pendulum
        (Phase 1: Transition)
        Phase 3: Rigid pendulum
        """
        # Partition
        N0 = self.opts.switch_stage
        N2 = self.opts.N - N0

        mp_ocp = AcadosMultiphaseOcp([N0, 1, N2])

        # ------- Phase 0: Spring-pendulum -------
        ocp_0 = AcadosOcp()
        model_0 = self._build_full_pendulum_model()  # 12D
        ocp_0.model = model_0

        # Nonlinear LS cost
        cost_expr_0 = self._cost_expr_full_pendulum(model_0)
        ocp_0.model.cost_y_expr = cost_expr_0
        ocp_0.cost.cost_type = "NONLINEAR_LS"

        # No terminal cost for phase 0, plays no roll anyways 
        W_0 = np.block([
            [self.opts.Q,             np.zeros((2, 2)),         np.zeros((2, 2))],
            [np.zeros((2, 2)),        self.opts.Q_acc_load,     np.zeros((2, 2))],
            [np.zeros((2, 2)),        np.zeros((2, 2)),         self.opts.R_reg]
        ])
        ocp_0.cost.W = W_0
        ocp_0.cost.yref = np.zeros(6)

        nx_0 = model_0.x.size()[0]
        ocp_0.constraints.x0 = np.zeros(nx_0)  # initial state only in phase 0

        # phi => index=2
        ocp_0.constraints.idxbx = np.array([2])
        ocp_0.constraints.ubx = np.array([self.opts.phi_max])
        ocp_0.constraints.lbx = np.array([self.opts.phi_min])
        # w => index=10,11
        ocp_0.constraints.idxbx = np.append(ocp_0.constraints.idxbx, [10, 11])
        ocp_0.constraints.ubx = np.append(ocp_0.constraints.ubx, [self.opts.w_max, self.opts.w_max])
        ocp_0.constraints.lbx = np.append(ocp_0.constraints.lbx, [self.opts.w_min, self.opts.w_min])
        # u => [dw1, dw2]
        ocp_0.constraints.idxbu = np.array([0, 1])
        ocp_0.constraints.ubu = np.array([self.opts.w_dot_max, self.opts.w_dot_max])
        ocp_0.constraints.lbu = np.array([self.opts.w_dot_min, self.opts.w_dot_min])

        self._set_ocp_solver_options(ocp_0)

        # ------- Phase 1: discrete transition-------
        ocp_1 = AcadosOcp()
        transition_model = self._build_transition_model()  # input is 12D
        ocp_1.model = transition_model

        transition_costs_expr = self._cost_expr_full_transition(transition_model)
        ocp_1.model.cost_y_expr = transition_costs_expr
        ocp_1.cost.cost_type = "NONLINEAR_LS"
        ocp_1.cost.W = self.opts.Q_trans
        ocp_1.cost.yref = np.zeros(2) # dimension=2 => [y_load_dd, z_load_dd]

        self._set_ocp_solver_options(ocp_1)

        # ------- Phase 2: Rigid Pendulum -------
        ocp_2 = AcadosOcp()
        model_2 = self._build_drone_rigid_pendulum_actuator_model(M_eff=self.opts.M + self.opts.m)
        ocp_2.model = model_2

        cost_expr_2 = self._cost_expr_drone_rigid_pendulum(model_2)  
        ocp_2.model.cost_y_expr = cost_expr_2
        ocp_2.cost.cost_type = "NONLINEAR_LS"

        # Terminal cost on first 2 comps => [y, z]
        ocp_2.model.cost_y_expr_e = ca.vertcat(cost_expr_2[0], cost_expr_2[1])
        ocp_2.cost.cost_type_e = "NONLINEAR_LS"

        W_2 = np.block([
            [self.opts.Q,             np.zeros((2, 2)),         np.zeros((2, 2))],
            [np.zeros((2, 2)),        self.opts.Q_acc_load,          np.zeros((2, 2))],
            [np.zeros((2, 2)),        np.zeros((2, 2)),          self.opts.R_reg]
        ])
        ocp_2.cost.W = W_2
        ocp_2.cost.W_e = self.opts.Q
        ocp_2.cost.yref = np.zeros(6)
        ocp_2.cost.yref_e = np.zeros(2)

        # no x0 for phase 2
        nx_2 = model_2.x.size()[0]
        ocp_2.constraints.x0 = np.zeros(nx_2)

        # phi => index=2
        ocp_2.constraints.idxbx = np.array([2])
        ocp_2.constraints.ubx = np.array([self.opts.phi_max])
        ocp_2.constraints.lbx = np.array([self.opts.phi_min])
        # w => index 8, 9
        ocp_2.constraints.idxbx = np.append(ocp_2.constraints.idxbx, [8, 9])
        ocp_2.constraints.ubx = np.append(ocp_2.constraints.ubx, [self.opts.w_max, self.opts.w_max])
        ocp_2.constraints.lbx = np.append(ocp_2.constraints.lbx, [self.opts.w_min, self.opts.w_min])
        # u => [dw1, dw2]
        ocp_2.constraints.idxbu = np.array([0, 1])
        ocp_2.constraints.ubu = np.array([self.opts.w_dot_max, self.opts.w_dot_max])
        ocp_2.constraints.lbu = np.array([self.opts.w_dot_min, self.opts.w_dot_min])

        self._set_ocp_solver_options(ocp_2)

        # Combine phases
        mp_ocp.set_phase(ocp_0, 0)
        mp_ocp.set_phase(ocp_1, 1)
        mp_ocp.set_phase(ocp_2, 2)

        self._set_ocp_solver_options(mp_ocp)

        # Overwrite some solver options that need to change for multi phase problems
        # Time steps (including discrete transition)
        step_sizes_list = list(self.opts.step_sizes[:N0]) + [1.0] + list(self.opts.step_sizes[N0:])
        mp_ocp.solver_options.time_steps = np.array(step_sizes_list)
        mp_ocp.solver_options.tf = sum(step_sizes_list)

        # Integrator choices
        mp_ocp.mocp_opts.integrator_type = [
            self.opts.integrator_type,  # phase 0: full pendulum
            "DISCRETE",                 # phase 1: transition
            "ERK",                      # phase 2: direct thrust
        ]

        self.total_ocp = mp_ocp


    ###########################################################################
    # Setting References 
    ###########################################################################

    def _set_reference_in_cost(self, pos_ref: np.ndarray):
        """
        y_ref = [y_target, z_target].
        We'll set [y_ref, z_ref, 0, 0, 0, 0, 0] (move to target point, but we want minimal load acceleration and inputs)
        """
        N = self.opts.N
        y_ref, z_ref = pos_ref[0], pos_ref[1]

        # For each shooting node: set a suitable yref dimension
        offset = 0
        for stage in range(N):
            if stage != 0 and stage == self.opts.switch_stage: # 0 stage means only 2d model and no switch
                offset += 1 # no reference update for witching stage, penalize large x3
                continue
            elif stage >= self.opts.switch_stage:
                y_ref_acados = np.array([y_ref, z_ref, 0.0, 0.0, 0.0, 0.0])
            else:
                y_ref_acados = np.array([y_ref, z_ref, 0.0, 0.0, 0.0, 0.0])
            self.acados_ocp_solver.set(stage + offset, "yref", y_ref_acados)     
        y_ref_N_acados = np.array([y_ref, z_ref])
        self.acados_ocp_solver.set(N+offset, "yref", y_ref_N_acados)
            

    ###########################################################################
    # Model-Building Routines
    ###########################################################################

    def _build_drone_rigid_pendulum_actuator_model(self, M_eff: float) -> AcadosModel:
        from drone.eom_2d_quadro_pend_explicit_casadi import eom_2d_quadro_pend_explicit 
        eom_func = eom_2d_quadro_pend_explicit
        x = ca.MX.sym("x", 10)
        u = ca.MX.sym("u", 2)

        params = ca.vertcat(
            self.opts.M,          # M
            self.opts.m,          # load mass
            self.opts.Ixx,        # Ixx
            self.opts.g,          # g
            self.opts.l0,         # rest length
            self.opts.c,          # rotor thrust constant
            self.opts.L_rot,      # half rotor distance
            0.0, 0.0, 0.0, 0.0    # external forces set to zero
        )

        xdot = eom_func(x, u, params)

        model = AcadosModel()
        model.name = f"drone_rigidpend_{self.timestamp}"
        model.x = x
        model.u = u
        model.xdot = ca.MX.sym("xdot", 10)
        model.f_expl_expr = xdot
        model.f_impl_expr = model.xdot - model.f_expl_expr
        return model

    def _build_full_pendulum_model(self) -> AcadosModel:
        import drone.eom_2d_quadro_springpend_explicit_casadi as eomfile

        x = ca.MX.sym("x", 12)
        u = ca.MX.sym("u", 2)

        # Build the params vector for the ODE function (external forces = 0)
        params = ca.vertcat(
            self.opts.M,          # M
            self.opts.m,          # load mass
            self.opts.Ixx,        # Ixx
            self.opts.g,          # g
            self.opts.k,          # spring stiffness
            self.opts.l0,         # rest length
            self.opts.c,          # rotor thrust constant
            self.opts.L_rot,      # half rotor distance
            0.0, 0.0, 0.0, 0.0    # external forces set to zero
        )

        # ODE from file
        x_dot_expr = eomfile.eom_2d_quadro_springpend_explicit(x, u, params)

        # Create the AcadosModel
        model = AcadosModel()
        model.name = f"drone_full_{self.timestamp}"
        model.x = x
        model.u = u
        nx = x.shape[0]
        model.xdot = ca.MX.sym("xdot", nx)
        model.f_expl_expr = x_dot_expr
        model.f_impl_expr = model.xdot - model.f_expl_expr

        # ----------------------------------------------------------
        # Build a CasADi expression for the load acceleration:
        #   y_load = y + r*sin(theta)
        #   z_load = z - r*cos(theta)
        # then differentiate twice w.r.t. time
        y_drone = x[0]
        z_drone = x[1]
        r_val   = x[3]
        th_val  = x[4]

        y_load = y_drone + r_val*ca.sin(th_val)
        z_load = z_drone - r_val*ca.cos(th_val)

        # first derivative
        dy_load_dx = ca.jacobian(y_load, x)
        y_load_dot = dy_load_dx @ x_dot_expr
        dz_load_dx = ca.jacobian(z_load, x)
        z_load_dot = dz_load_dx @ x_dot_expr

        # second derivative
        dy_load_dot_dx = ca.jacobian(y_load_dot, x)
        y_load_ddot = dy_load_dot_dx @ x_dot_expr
        dz_load_dot_dx = ca.jacobian(z_load_dot, x)
        z_load_ddot = dz_load_dot_dx @ x_dot_expr

        # Store as a Function => model.load_acc_func( x_12 ) => [y_load_dd, z_load_dd]
        # We'll need this for the transition cost.
        self.load_acc_func = ca.Function(
            "load_acc_func",
            [model.x, model.u], [y_load_ddot, z_load_ddot],
            ["x_in", "u_in"], ["y_load_dd", "z_load_dd"]
        )

        return model

    def _build_transition_model(self) -> AcadosModel:
        """
        Discrete map from 12D => 10D. Neglect spring states.
        """
        x_in = ca.SX.sym("x_in", 12)
        disc_expr = ca.vertcat(
            x_in[0],  # y
            x_in[1],  # z
            x_in[2],  # phi
            x_in[4],  # theta
            x_in[5],  # y_dot
            x_in[6],  # z_dot
            x_in[7],  # phi_dot
            x_in[9],  # theta_dot
            x_in[10], # w_1
            x_in[11], # w_2
        )

        model = AcadosModel()
        model.name = f"transition_{self.timestamp}"
        model.x = x_in
        model.disc_dyn_expr = disc_expr
        return model

    ###########################################################################
    # Cost Expressions
    ###########################################################################

    def _cost_expr_drone_rigid_pendulum(self, model: AcadosModel) -> ca.SX:
        """
        cost_y_expr = [y, z, y_load_dd, z_load_dd, dw1, dw2],
        where y_load = y + l0*sin(theta) and z_load = z - l0*cos(theta).
        This function computes the second derivative of the load position
        (i.e. the load acceleration) and concatenates it with the control inputs.
        """
        x_sym = model.x
        u_sym = model.u  # [dw1, dw2]
        # Parse states: x = [y, z, phi, theta, y_dot, z_dot, phi_dot, theta_dot, w1, w2]
        y_ = x_sym[0]
        z_ = x_sym[1]
        phi_ = x_sym[2]
        theta_ = x_sym[3]

        # Build load position: fixed pendulum length (l0)
        y_load = y_ + self.opts.l0 * ca.sin(theta_)
        z_load = z_ - self.opts.l0 * ca.cos(theta_)

        # Compute first derivative of load position
        xdot_expr = model.f_expl_expr
        dy_load_dx = ca.jacobian(y_load, x_sym)
        y_load_dot = dy_load_dx @ xdot_expr
        dz_load_dx = ca.jacobian(z_load, x_sym)
        z_load_dot = dz_load_dx @ xdot_expr

        # Compute second derivative (acceleration) of load position
        dy_load_dot_dx = ca.jacobian(y_load_dot, x_sym)
        y_load_dd = dy_load_dot_dx @ xdot_expr
        dz_load_dot_dx = ca.jacobian(z_load_dot, x_sym)
        z_load_dd = dz_load_dot_dx @ xdot_expr

        cost_expr = ca.vertcat(
            y_,         # Drone y-position
            z_,         # Drone z-position
            ca.sign(y_load_dd) * ca.fmin(1e12, ca.fabs(y_load_dd)),  # Load acceleration in y-direction
            ca.sign(z_load_dd) * ca.fmin(1e12, ca.fabs(z_load_dd)),  # Load acceleration in z-direction
            u_sym[0],   # Control: dw1
            u_sym[1]    # Control: dw2
        )
        return cost_expr

    def _cost_expr_full_pendulum(self, model: AcadosModel) -> ca.SX:
        """
        cost_y_expr = [ y, z, y_load_dd, z_load_dd, dw1, dw2]
        We'll reuse the symbolic function that we stored in 'self.load_acc_func':
           [y_load_dd, z_load_dd] = self.load_acc_func(x).
        """
        x_sym = model.x
        y_drone = x_sym[0]
        z_drone = x_sym[1]

        # compute [y_load_ddot, z_load_ddot] using the stored function
        load_acc = self.load_acc_func(x_sym, model.u)  # => [y_load_dd, z_load_dd]
        y_load_ddot = load_acc[0]
        z_load_ddot = load_acc[1]

        u_1 = model.u[0]
        u_2 = model.u[1]

        cost_expr = ca.vertcat(
            y_drone, 
            z_drone, 
            ca.sign(y_load_ddot) * ca.fmin(1e12, ca.fabs(y_load_ddot)), 
            ca.sign(z_load_ddot) * ca.fmin(1e12, ca.fabs(z_load_ddot)), 
            u_1, 
            u_2
            )
        return cost_expr
    
    def _cost_expr_full_transition(self, model: AcadosModel) -> ca.SX:
        """
        Unused as Q_trans is all zeros, but contains a draft of transition costs
        """
        x_sym = model.x

        # compute [y_load_ddot, z_load_ddot] using the stored function
        load_acc = self.load_acc_func(x_sym, np.zeros(2))  # => [y_load_dd, z_load_dd]
        y_load_ddot = load_acc[0]
        z_load_ddot = load_acc[1]

        cost_expr = ca.vertcat(
            ca.sign(y_load_ddot) * ca.fmin(1e12, ca.fabs(y_load_ddot)), 
            ca.sign(z_load_ddot) * ca.fmin(1e12, ca.fabs(z_load_ddot))
            )
        return cost_expr


    ###########################################################################
    # Solver Options
    ###########################################################################

    def _set_ocp_solver_options(self, ocp: AcadosOcp):
        ocp.solver_options.nlp_solver_type = self.opts.nlp_solver_type
        ocp.solver_options.qp_solver = self.opts.qp_solver
        ocp.solver_options.integrator_type = self.opts.integrator_type
        ocp.solver_options.nlp_solver_max_iter = 300
        ocp.solver_options.tol = 1e-6
        ocp.solver_options.qp_tol = 1e-2*ocp.solver_options.tol
        ocp.solver_options.print_level = 1
        ocp.solver_options.qp_solver_iter_max = 500
        ocp.solver_options.hessian_approx = "EXACT"#"GAUSS_NEWTON"
        ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        ocp.solver_options.levenberg_marquardt = self.opts.levenberg_marquardt # regularize Hessian stongly due to conditioning
        ocp.solver_options.hpipm_mode = "ROBUST" # problem seems a bit ill conditioned, try robust mode
        # ocp.solver_options.tf or time_steps must be set outside as appropriate.

    ###########################################################################
    # AcadosSimSolver for the Full Model Only
    ###########################################################################

    def _create_sim_full_model(self):
        """
        Create a single AcadosSimSolver for the full pendulum model for potential
        closed-loop simulation or debugging.
        """
        sim_full = AcadosSim()
        model_full = self._build_full_pendulum_model()
        sim_full.model = model_full

        sim_full.solver_options.T = self.opts.step_sizes[0]
        sim_full.solver_options.integrator_type = self.opts.integrator_type

        solvers_dir = get_dir("solvers") / "drone"
        solvers_dir.mkdir(parents=True, exist_ok=True)
        sim_full.code_export_directory = str(solvers_dir / f"c_generated_code_sim_full_{self.timestamp}")
        sim_json = solvers_dir / f"acados_sim_full_{self.timestamp}.json"
        sim_solver = AcadosSimSolver(
            sim_full, json_file=str(sim_json)
        )
        self.sim_solver_full = sim_solver
