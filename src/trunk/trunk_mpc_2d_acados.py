from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver, AcadosMultiphaseOcp
import casadi as ca
import numpy as np
from scipy.linalg import block_diag
from trunk.ODE_chains.eom_system_2_casadi import eom_2
from trunk.ODE_chains.eom_system_4_casadi import eom_4
from trunk.ODE_chains.eom_system_8_casadi import eom_8
from trunk.ODE_chains.eom_system_16_casadi import eom_16
from trunk.ODE_chains.ODE_utils import get_ode_params_dict, params_dict_to_list
from trunk.trunk_utils import compute_q_casadi
from utils_shared import get_dir
from typing import List, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

@dataclass
class TrunkMPC2DOptions:
    nlp_solver_type: str  
    n: List[int]          
    N_list: List[int]     
    n_high: Optional[int] = None  
    dt: float = 0.004
    print_level: int = 0
    ub_x_ee: List[float] = field(default_factory=lambda: [1e15, 1e15])
    lb_x_ee: List[float] = field(default_factory=lambda: [-1e15, -1e15])
    levenberg_marquardt: float = 0.0
    qp_solver: str = 'FULL_CONDENSING_HPIPM'

class TrunkMPC2D(ABC):
    """
    This class (and its subclasses) implement the trunk MPC in acados.
    Initial trials included not only the point-mass model as an approximate model, but also the 
    Settings can be adjusted in the TrunkMPC2DOptions object that is passed during instantiation.
    Supports model schedules with more than 2 models. The models, step sizes and transitions are controlled via n, N_list, and dt
    n: List of used chain models, for instance [16, 8, 'p'] means that we have three phases switching from the 16 link model to the 8 link model and then to the point mass model.
    dt: list of integration step sizes
    N_list: determines the number of shooting intervals in each phase for instance [10,10,10] means that all three models are used for 10 shooting intervals each.
    """
    def __init__(self, trunk_mpc_2d_options: TrunkMPC2DOptions) -> None:

        # expand passed options
        self.n = trunk_mpc_2d_options.n
        self.n_string = "_".join(map(str, self.n)) 
        self.n_last = self.n
        self.N_list = trunk_mpc_2d_options.N_list
        self.N_list_str = "_".join(map(str, self.N_list))  # Convert N_list to a string with underscores
        self.n_high = trunk_mpc_2d_options.n_high
        self.nlp_solver_type = trunk_mpc_2d_options.nlp_solver_type
        if isinstance(trunk_mpc_2d_options.dt, list):
            self.dt = trunk_mpc_2d_options.dt
        else:
            self.dt = [trunk_mpc_2d_options.dt]*sum(self.N_list)
        self.dt_string = str(sum(self.dt)).replace('.', '')
        self.print_level = trunk_mpc_2d_options.print_level
        self.ub_x_ee = trunk_mpc_2d_options.ub_x_ee
        self.lb_x_ee = trunk_mpc_2d_options.lb_x_ee
        self.x_has_constraint = not (self.ub_x_ee[0] == 1e15 and self.lb_x_ee[0] == -1e15)
        self.y_has_constraint = not (self.ub_x_ee[1] == 1e15 and self.lb_x_ee[1] == -1e15)
        self.levenberg_marquardt = trunk_mpc_2d_options.levenberg_marquardt
        self.qp_solver = trunk_mpc_2d_options.qp_solver

        self.multi_phase = False
        if isinstance(self.n, list): 
            if len(self.n) >= 2:
                self.multi_phase = True
                self.n_phases = len(self.n)
                self.N_list_with_transitions = self.add_transitions_to_N_list()
                self.n_last = self.n[-1]
            else:
                self.n = self.n[0]
                self.n_last = self.n
                self.n_phases = 1

        # save ode functions as attributes
        self.eom = {
            "2": eom_2,
            "4": eom_4,
            "8": eom_8,
            "16": eom_16
        }

        # load ode params
        self.ode_params_dict = get_ode_params_dict()

        # get acados ocp and generate solver
        self.ocp = self.export_ocp()

        # Use the timestamp in the file name
        self.solvers_dir = get_dir("solvers") / "trunk"
        self.solvers_dir.mkdir(parents=True, exist_ok=True)
        json_stem = f"acados_ocp_solver_chain_{self.n_string}_{self.N_list_str}_{self.dt_string}_{self.qp_solver}"
        self.ocp.code_export_directory = str(self.solvers_dir / f"c_generated_code_{json_stem}")
        self.acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file=str(self.solvers_dir / f"{json_stem}.json"))

        self.acados_sim_solver = self.create_sim_solver(self.multi_phase)

    def export_ocp(self):
        if self.multi_phase:
            ocp = AcadosMultiphaseOcp(N_list=self.N_list_with_transitions)
            ocp.solver_options.tf = sum(self.dt) + 1.0*(self.n_phases-1) # same total time horizon plus transitions

            time_steps = []
            scaling_factor = 1.0 # scaling factor to reduce costs for later parts of the horizon
            dt_pointer = 0 # Index tracking for extracting the dt values corresponding to the different phases
            for phase in range(self.n_phases-1):
                n_current = self.n[phase]
                n_next = self.n[phase+1]

                # get ocps
                phase_ocp = self.export_chain_ocp(n_current)
                transition_ocp = self.export_transition_ocp(n_current, n_next)

                # scale costs for later ocps
                phase_ocp.cost.W = scaling_factor**phase*phase_ocp.cost.W  # Combined weight matrix

                # set ocps for different phases
                ocp.set_phase(phase_ocp, 2*phase)
                ocp.set_phase(transition_ocp, 2*phase+1)

                num_timesteps = self.N_list[phase]
                dt_phase = self.dt[dt_pointer : dt_pointer + num_timesteps]
                dt_pointer += num_timesteps  # Move pointer forward
                time_steps.extend(dt_phase)
                time_steps.append(1.0) # for transition stage

            # last phase
            last_phase_ocp, last_integrator = self.export_last_ocp()
            last_phase_ocp.cost.W = scaling_factor**self.n_phases*last_phase_ocp.cost.W
            last_phase_ocp.cost.W_e = scaling_factor**self.n_phases*last_phase_ocp.cost.W_e
            ocp.set_phase(last_phase_ocp, 2*(self.n_phases-1))
            time_steps.extend(self.dt[dt_pointer:])
            
            # set mocp solver options
            ocp.mocp_opts.integrator_type = ['IRK', 'DISCRETE']*(self.n_phases-1) + [last_integrator]

            # save total number stages
            self.N = sum(self.N_list_with_transitions)

        else:
            ocp, _ = self.export_last_ocp()
            self.N = self.N_list[0]
            time_steps = self.dt
            ocp.dims.N = self.N  # Number of shooting intervals (time steps)
            ocp.solver_options.tf = sum(self.dt)# Prediction horizon (seconds)

        # set ocp options 
        ocp.solver_options.qp_solver = self.qp_solver 
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.nlp_solver_type = self.nlp_solver_type
        ocp.solver_options.time_steps = np.array(time_steps)
        ocp.solver_options.nlp_solver_max_iter = 150
        ocp.solver_options.print_level = self.print_level
        ocp.solver_options.levenberg_marquardt = self.levenberg_marquardt
        ocp.solver_options.nlp_solver_tol_comp = 1e-7
        ocp.solver_options.nlp_solver_tol_eq = 1e-7
        ocp.solver_options.nlp_solver_tol_ineq = 1e-7
        ocp.solver_options.nlp_solver_tol_stat = 1e-7  

        return ocp 

    def add_transitions_to_N_list(self):
        N_list_with_transitions = []
        for number in self.N_list:
            N_list_with_transitions.append(number) 
            N_list_with_transitions.append(1)    # Add 1.0 after each number for the transitions
        N_list_with_transitions.pop()  # Remove the last appended 1.0 since it's unnecessary
        return N_list_with_transitions    

    def create_sim_solver(self, multi_phase, stage=0):
        acados_sim = AcadosSim()
        acados_sim.solver_options.T = self.dt[0] # set time horizon to a single time step
        if multi_phase:
            acados_sim.model = self.ocp.model[stage]
            acados_sim.solver_options.integrator_type = self.ocp.mocp_opts.integrator_type[stage]
        else:
            acados_sim.model = self.ocp.model
            acados_sim.solver_options.integrator_type = self.ocp.solver_options.integrator_type
        solvers_dir = getattr(self, "solvers_dir", get_dir("solvers") / "trunk")
        solvers_dir.mkdir(parents=True, exist_ok=True)
        json_stem = f"acados_sim_solver_chain_{self.n_string}_{self.N_list_str}_{stage}_{self.dt_string}_{self.qp_solver}"
        acados_sim.code_export_directory = str(solvers_dir / f"c_generated_code_{json_stem}")
        return AcadosSimSolver(acados_sim, json_file=str(solvers_dir / f"{json_stem}.json"))

    def set_initial_state(self, x0):
        self.acados_ocp_solver.set(0, "lbx", x0)
        self.acados_ocp_solver.set(0, "ubx", x0)

    def get_planned_trajectory(self):
        traj_x, traj_u = [], []
        for i in range(self.N):  # N+1 to include the final step
            x_i = self.acados_ocp_solver.get(i, "x")
            u_i = self.acados_ocp_solver.get(i, "u")
            traj_x.append(x_i)
            traj_u.append(u_i)
        x_N = self.acados_ocp_solver.get(self.N, "x")
        traj_x.append(x_N)
        return traj_x, traj_u

    def solve_ocp(self):
        status = self.acados_ocp_solver.solve()
        #self.ocp_solver.dump_last_qp_to_json() # activate dumping for debugging purposes
        if status != 0:
            print("Acados returned solver status ", status)
            #raise RuntimeError(f"OCP solver failed with status {status}.")
        u0  = self.acados_ocp_solver.get(0, "u")
        total_time = self.acados_ocp_solver.get_stats('time_tot') 
        n_iters = self.acados_ocp_solver.get_stats('sqp_iter')
        return u0, total_time, n_iters
    
    def set_initial_guess(self, guess_x, guess_u):
        for stage in range(self.N):
            self.acados_ocp_solver.set(stage, 'x', guess_x[stage])
            self.acados_ocp_solver.set(stage, 'u', guess_u[stage])
        self.acados_ocp_solver.set(self.N, 'x', guess_x[stage])
    
    @abstractmethod
    def export_chain_ocp(self, n):
        pass
    
    def export_chain_model(self, n):

        # Define symbolic variables
        q = ca.MX.sym('q', n)  # generalized coodinates
        q_dot = ca.MX.sym('q_dot', n)  # generalized coodinates
        x = ca.vertcat(q, q_dot) # state
        u = ca.MX.sym('u', 2)  # Control input: velocity/force
        model_name = f'model_{n}_links_{self.n_string}_{self.N_list_str}_{self.dt_string}_{self.qp_solver}'  # Name the model
        u_expanded = ca.vertcat( # expand u to actuate all joints
            ca.repmat(u[0], n // 2, 1),  # Repeat u[0] n_half times
            ca.repmat(u[1], n // 2, 1)  # Repeat u[1] n_half times
        )

        xdot = ca.MX.sym('xdot', 2*n)
        q_ddot = xdot[n:]

        # Create the AcadosModel 
        model = AcadosModel()
        model.name = model_name
        model.x = x  # Define the state
        model.q = q  # Also store the generalized coordinates (part of the state) 
        model.q_dot = q_dot
        model.u = u  # Define the control input
        model.xdot = xdot
        
        f_impl_expr_q_dot = xdot[:n] - q_dot
        params_list = params_dict_to_list(self.ode_params_dict[f"{n}"])
        f_impl_expr_q_ddot = self.eom[f"{n}"](q, q_dot, q_ddot, u_expanded, params_list)
        model.f_impl_expr = ca.vertcat(f_impl_expr_q_dot, f_impl_expr_q_ddot)  # Implicit dynamics 

        return model
    
    def export_transition_model_chain(self, n_high_dim, n_low_dim):
        model = AcadosModel()
        model.name = f"transition_model_{n_high_dim}_{n_low_dim}_{self.n_string}_{self.N_list_str}_{self.dt_string}_{self.qp_solver}"
        q_high_dim = ca.MX.sym('q', n_high_dim)  # generalized coodinates
        q_dot_high_dim = ca.MX.sym('q_dot', n_high_dim)  # generalized coodinates
        model.x = ca.vertcat(q_high_dim, q_dot_high_dim) # state
        model.u = ca.SX.sym('u', 0, 0)
        
        # map high dimensional q and q_dots to low dimensional ones
        link_length_high_dim = 2*self.ode_params_dict[f"{n_high_dim}"]["l"]
        link_length_low_dim = 2*self.ode_params_dict[f"{n_low_dim}"]["l"]
        q_low_dim = compute_q_casadi(q_high_dim, link_length_high_dim, link_length_low_dim, n_low_dim)
        jacobian_q_low_dim_q_high_dim = ca.jacobian(q_low_dim, q_high_dim) 
        q_dot_low_dim = jacobian_q_low_dim_q_high_dim @ q_dot_high_dim  # Velocity mapping for 8-link
        
        model.disc_dyn_expr = ca.vertcat(q_low_dim, q_dot_low_dim)
        return model
    
    def export_transition_ocp(self, n_current, n_next):
        return self.export_transition_ocp_chain(n_current, n_next)
    
    def export_last_ocp(self):
        last_phase_ocp = self.export_chain_ocp(self.n_last)
        last_integrator = 'IRK'
        return last_phase_ocp, last_integrator
    
    def export_transition_ocp_chain(self, n_high_dim, n_low_dim):
        ocp = AcadosOcp()
        ocp.model = self.export_transition_model_chain(n_high_dim, n_low_dim)
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.model.cost_y_expr = ocp.model.x
        ocp.cost.W = np.diag([0, 0]*n_high_dim)
        ocp.cost.yref = np.array([0, 0]*n_high_dim)
        return ocp

    def set_reference_trajectory(self, t0):
        if self.multi_phase:
            # Multi-phase case
            index_offset = 0
            t_eval = t0
            t_update_counter = 0
            for phase, N_phase in enumerate(self.N_list):
                for i in range(N_phase):
                    y_ref = self.get_y_ref(t_eval, t0, n_phase=self.n[phase])
                    self.acados_ocp_solver.set(index_offset + i, "yref", y_ref)
                    self.set_additional_ocp_parameters(index_offset + i, t_eval, t0, phase)
                    t_eval += self.dt[t_update_counter]
                    t_update_counter += 1

                # Update offset and move to the next phase
                index_offset += N_phase + 1  # Skip transition stage

            # Set terminal reference for the phase
            last_idx = index_offset - 1
            y_ref_N = self.get_y_ref(t_eval, t0, n_phase=self.n[-1], terminal=True)
            self.acados_ocp_solver.set(last_idx, "yref", y_ref_N)   
            self.set_additional_ocp_parameters(last_idx, t_eval, t0, phase, terminal=True) 
        else:
            # Single-phase case
            N = self.ocp.dims.N
            t_eval = t0
            for i in range(N):
                y_ref = self.get_y_ref(t_eval, t0, n_phase=self.n)
                self.acados_ocp_solver.set(i, "yref", y_ref)
                t_eval += self.dt[i]
            
            # Set terminal reference
            y_ref_N = self.get_y_ref(t_eval, t0, n_phase=self.n, terminal=True)
            self.acados_ocp_solver.set(N, "yref", y_ref_N)
            self.set_additional_ocp_parameters(N, t_eval, t0, terminal=True)

    @abstractmethod
    def get_y_ref(self, t, t0, n_phase=None, terminal=False):
        pass
        
    def forward_kinematics_casadi(self, q, n):
        l = get_ode_params_dict()[f"{n}"]["l"] # note that l in params dict corresponds to half length
        x_pos, y_pos, q_tot = 0, 0, 0
        for i in range(n):
            q_tot += q[i]
            x_pos += 2*l*ca.sin(q_tot)
            y_pos -= 2*l*ca.cos(q_tot)
        return x_pos, y_pos 
    
    def set_additional_ocp_parameters(self , stage, t_stage, t0, phase=None, terminal=False):
        """placeholder function to adaptively set other ocp parameters with the set reference function"""
        pass

class TrunkMPC2DEETracking(TrunkMPC2D):
    def __init__(self, Phi_t, Phi_dot_t, trunkMPC2DOptions):
        self.ub_x_ee = trunkMPC2DOptions.ub_x_ee
        
        # reference trajecory to track
        self.Phi_t = Phi_t
        self.Phi_dot_t = Phi_dot_t

        super().__init__(trunkMPC2DOptions)
        
    def get_planned_ee_trajectory(self):
        traj_x, _ = self.get_planned_trajectory()
        if self.multi_phase and self.n[-1] == 'p':
            ee_traj_part_1 = np.array([np.array(self.forward_kinematics_casadi(q, len(q) // 2)) for q in traj_x[:-(self.N_list[-1]+1)]])
            ee_traj_part_2 = np.array(traj_x[-self.N_list[-1]:])[:, :2]
            ee_traj = np.vstack((ee_traj_part_1, ee_traj_part_2))
        else:
            ee_traj = np.array([np.array(self.forward_kinematics_casadi(q, len(q) // 2)) for q in traj_x])
        return ee_traj
    
    def export_transition_ocp(self, n_current, n_next):
        if n_next == 'p':
            transition_ocp = self.export_transition_ocp_point_mass(n_current)
        else:
            transition_ocp = self.export_transition_ocp_chain(n_current, n_next)
        return transition_ocp
    
    def export_last_ocp(self):
        if self.n_last == 'p':
            last_phase_ocp = self.export_ocp_point_mass()
            last_integrator = 'ERK'
        else:
            last_phase_ocp = self.export_chain_ocp(self.n_last)
            last_integrator = 'IRK'
        return last_phase_ocp, last_integrator
    
    def export_chain_ocp(self, n):
        ocp = AcadosOcp()
        
        # generate acados model
        model = self.export_chain_model(n) 
        ocp.model = model  

        # dimensions
        nx = model.x.shape[0]
        nu = model.u.shape[0]
        n_ee = 2 # number of endeffector coordinates to consider, x and y as we operate in 2d case
        ny = 2*n_ee + nu 
        ny_e = 2*n_ee 

        # quadratic costs
        Q = np.diag([4.0, 4.0, 0.05, 0.05])  # State cost weight
        R = np.diag([0.015, 0.015])  # Control cost weight
        ocp.cost.Q = Q
        ocp.cost.R = R
        ocp.cost.cost_type = 'NONLINEAR_LS'  # Linear least squares cost
        ocp.cost.cost_type_e = 'NONLINEAR_LS'  # Terminal cost type 
        ocp.cost.W = block_diag(Q, R)  # Combined weight matrix
        ocp.cost.W_e = Q # Terminal cost (penalize final state)

        # dummy reference trajectories (all zeros for now), need to be overwritten with the set_reference_trajectory function
        ocp.cost.yref = np.zeros((ny,)) # Reference: [ee_ref, u_ref]
        ocp.cost.yref_e = np.zeros((ny_e,))   # Terminal reference for now: [ee_ref]

        # define y_expr for nonlinear LS
        x_ee, y_ee = self.forward_kinematics_casadi(model.q, n)
        ee_pos = ca.vertcat(x_ee, y_ee)

        # Compute the end-effector velocity (ee_pos_dot) using CasADi's AD
        x_ee_dot = ca.jacobian(x_ee, model.q) @ model.q_dot  # Chain rule: dx/dq * q_dot
        y_ee_dot = ca.jacobian(y_ee, model.q) @ model.q_dot  # Chain rule: dy/dq * q_dot
        ee_pos_dot = ca.vertcat(x_ee_dot, y_ee_dot)  # Combine velocities

        # Update the cost expressions to include ee_pos_dot
        ocp.model.cost_y_expr = ca.vertcat(ee_pos, ee_pos_dot, model.u)
        ocp.model.cost_y_expr_e = ca.vertcat(ee_pos, ee_pos_dot)  # Terminal cost remains only on ee_pos

        if self.x_has_constraint or self.y_has_constraint:
            ocp = self._add_half_space_constraints(ocp, ee_pos)

        # initial state constraint 
        ocp.constraints.x0 = np.zeros((nx,1))

        # Solver settings, rest is set in init function
        ocp.solver_options.integrator_type = 'IRK'  

        return ocp
    
    def _add_half_space_constraints(self, ocp, ee_pos):
        if self.x_has_constraint and self.y_has_constraint:
            # Both x and y are constrained, so use the full end-effector position
            ocp.model.con_h_expr = ee_pos  
            ocp.model.con_h_expr_e = ee_pos  

            ocp.constraints.lh = np.array(self.lb_x_ee)  
            ocp.constraints.lh_e = np.array(self.lb_x_ee)

            ocp.constraints.uh = np.array(self.ub_x_ee)  
            ocp.constraints.uh_e = np.array(self.ub_x_ee)

        elif self.x_has_constraint:
            # Only x is constrained
            ocp.model.con_h_expr = ee_pos[0]  
            ocp.model.con_h_expr_e = ee_pos[0]  

            ocp.constraints.lh = np.array([self.lb_x_ee[0]])  
            ocp.constraints.lh_e = np.array([self.lb_x_ee[0]])

            ocp.constraints.uh = np.array([self.ub_x_ee[0]])  
            ocp.constraints.uh_e = np.array([self.ub_x_ee[0]])

        elif self.y_has_constraint:
            # Only y is constrained
            ocp.model.con_h_expr = ee_pos[1]  
            ocp.model.con_h_expr_e = ee_pos[1]  

            ocp.constraints.lh = np.array([self.lb_x_ee[1]])  
            ocp.constraints.lh_e = np.array([self.lb_x_ee[1]])

            ocp.constraints.uh = np.array([self.ub_x_ee[1]])  
            ocp.constraints.uh_e = np.array([self.ub_x_ee[1]])
        return ocp
    
    def export_model_point_mass(self):
        # state
        x = ca.MX.sym('x')       
        y = ca.MX.sym('y')       
        vx = ca.MX.sym('vx')     
        vy = ca.MX.sym('vy')     

        # controls
        ax = ca.MX.sym('ax')     
        ay = ca.MX.sym('ay') 

        # Combine states and controls into vectors
        states = ca.vertcat(x, y, vx, vy)
        controls = ca.vertcat(ax, ay)

        # dx/dt = [vx, vy, ax, ay]
        f_expl = ca.vertcat(vx, vy, ax, ay)

        # Create the ACADOS model
        model = AcadosModel()
        model.name = f"point_mass_{self.n_string}_{self.N_list_str}_{self.dt_string}_{self.qp_solver}"
        model.xdot = ca.MX.sym('xdot', 4)   

        # Assign states, controls, and dynamics
        model.x = states          # State vector
        model.ee_pos = ca.vertcat(x, y)
        model.u = controls        # Control vector
        model.f_expl_expr = f_expl  # Explicit dynamics
        model.f_impl_expr = model.xdot - f_expl  # Implicit dynamics (if needed)

        return model

    def export_transition_model_point_mass(self, n_prior):
        model = AcadosModel()
        model.name = f"transition_model_point_mass_{self.n_string}_{self.N_list_str}_{self.dt_string}_{self.qp_solver}"
        q_prior = ca.MX.sym('q', n_prior)  # generalized coodinates
        q_dot_prior = ca.MX.sym('q_dot', n_prior)  # generalized coodinates
        model.x = ca.vertcat(q_prior, q_dot_prior) # state
        model.u = ca.SX.sym('u', 0, 0)
        
        # map high dimensional q and q_dots to endeffector position
        x_pos_ee, y_pos_ee = self.forward_kinematics_casadi(q_prior, n_prior)   
        x_ee_dot = ca.jacobian(x_pos_ee, q_prior) @ q_dot_prior # Chain rule: dx/dq * q_dot
        y_ee_dot = ca.jacobian(y_pos_ee, q_prior) @ q_dot_prior  # Chain rule: dy/dq * q_dot    
        model.disc_dyn_expr = ca.vertcat(x_pos_ee, y_pos_ee, x_ee_dot, y_ee_dot)
        return model
    
    def export_transition_ocp_point_mass(self, n_prior):
        ocp = AcadosOcp()
        ocp.model = self.export_transition_model_point_mass(n_prior)
        nx = ocp.model.x.shape[0]
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.model.cost_y_expr = ocp.model.x
        ocp.cost.W = np.diag([0]*nx)
        ocp.cost.yref = np.array([0]*nx) 

        return ocp
    
    def export_ocp_point_mass(self):
        ocp = AcadosOcp()
        ocp.model = self.export_model_point_mass()
        nx = ocp.model.x.shape[0]
        nu = ocp.model.u.shape[0]
        ny = nx + nu 
        ny_e = nx
        
        Q = np.diag([4.0, 4.0, 0.05, 0.05])  # State cost weight
        R = np.diag([0.00001, 0.00001])  # Control cost weight,
        ocp.cost.cost_type = 'NONLINEAR_LS'  # Linear least squares cost
        ocp.cost.cost_type_e = 'NONLINEAR_LS'  # Terminal cost type 
        ocp.model.cost_y_expr = ca.vertcat(ocp.model.x, ocp.model.u)
        ocp.model.cost_y_expr_e = ocp.model.x
        ocp.cost.W = block_diag(Q, R)  # Combined weight matrix
        ocp.cost.W_e = Q # Terminal cost (penalize final state)

        ocp.cost.yref = np.zeros((ny,)) 
        ocp.cost.yref_e = np.zeros((ny_e,)) 

        # constraints
        if self.x_has_constraint or self.y_has_constraint:
            ocp = self._add_half_space_constraints(ocp, ocp.model.ee_pos)
        
        return ocp
    
    def get_y_ref(self, t, t0, n_phase=None, terminal=False):
        theta_ref = np.array(self.Phi_t(t)).squeeze()
        theta_dot_ref = np.array(self.Phi_dot_t(t)).squeeze()
        y_ref = [theta_ref[0], theta_ref[1], theta_dot_ref[0], theta_dot_ref[1]]
        if not terminal:
            y_ref += [0, 0]
        return np.array(y_ref)
