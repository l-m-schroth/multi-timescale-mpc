"""
This file compares the derived ODE using sympy with the mujoco, to make sure the ODe is correct
"""

import numpy as np
from scipy.optimize import root
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import importlib
import os
from trunk.ODE_chains.ODE_utils import get_ode_params_dict, params_dict_to_list
import mujoco
import sys
from utils_shared import get_dir

# Dynamically add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trunk.trunk_utils import simulate

# NOTE: current integration approach with root scipy.optimize root in the loop is not very robust. This should be replaced witha "poper" implicit scheme fo stiff ODEs.
# However, as this script is just to verify correctness of the derived ODEs, the current implementation sufficies.

# --- User-defined parameters ---
n = 4  # Number of generalized coordinates (must match the saved function)
t_end = 2
t_span = (0, t_end)  # Time range for integration
dt = 0.01 # implicit integration works with much larger step sizes, system is dynamically stiff.
dt_mujoco = 0.0001 #try f.ex 0.0005, integration fails with RK4
N_steps = int(t_end / dt) + 1
N_steps_mujoco = int(t_end / dt_mujoco) + 1
t_eval = np.linspace(0, t_end, N_steps)  # TimeZ points for evaluation
t_eval_mujoco = np.linspace(0, t_end, N_steps_mujoco)  # TimeZ points for evaluation

u_1 = -0.02#.06
u_2 = 0.02#

params_dict = get_ode_params_dict()
params = params_dict_to_list(params_dict[f"{n}"])  #[9.81, 1.0, 1.0, 0.1, 10.0, 0.5] works, but real params seem to be hard to work with ## g, m, l, Iz, c, d
u = [u_1]*(n // 2) + [u_2]*(n // 2)

# Initial conditions: q, q_dot
q0 = np.array([-np.pi/64]*(n // 2) + [-np.pi/32]*(n // 2))  # Random initial positions
q_dot0 = np.zeros(n)  # Zero initial velocities
y0 = np.concatenate((q0, q_dot0))  # Combined state vector

# --- Load the corresponding eom_system_n function ---
module_name = f"eom_system_{n}"  # File name should be eom_system_n.py
eom_module = importlib.import_module(module_name)
eom = getattr(eom_module, f"eom_{n}")  # Dynamically import the correct eom function

# --- Define the implicit ODE solver ---
q_ddot_previous = np.zeros(n)
def solve_implicit_ode(t, y):
    """Solve the implicit ODE by determining q_ddot at each step."""
    global q_ddot_previous  # Declare global to modify the variable outside the function
    n = len(y) // 2
    q = y[:n]
    q_dot = y[n:]

    # Function to find q_ddot such that eom(t, q, q_dot, q_ddot, params) == 0
    def implicit_eq(q_ddot):
        return eom(q, q_dot, q_ddot, u, params)

    # Solve for q_ddot using root-finding
    result = root(implicit_eq, q_ddot_previous, tol=1e-8)  # Initial guess for q_ddot
    if not result.success:
        raise RuntimeError(f"Root-finding failed at time {t} with message: {result.message}")
    q_ddot = result.x
    q_ddot_previous = q_ddot

    # Return the state derivative: [q_dot, q_ddot]
    return np.concatenate((q_dot, q_ddot))

# --- Integrate the system ---
solution = solve_ivp(
    solve_implicit_ode, t_span, y0, t_eval=t_eval, method="Radau"
)

# --- simulate the actual mujoco system for comparison ---
dir = get_dir("src/trunk")
xml_path = dir / "archive" / "models_mujoco" / f"chain_{n}_links_expanded.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
model.opt.timestep = dt_mujoco
model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
data = mujoco.MjData(model)
ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "endeffector")

ee_positions, qpos, qvel = simulate(model, data, t_end, ee_site_id, q0, q_dot0, u_controls=[u_1, u_2])

# Convert qpos to a NumPy array to enable slicing
qpos = np.array(qpos)

# --- Generate a nice plot to compare if the ODE is correct and aligns with MuJoCo ---
plt.figure(figsize=(10, 6))

# Plot generalized coordinates over time for both ODE and MuJoCo
for i in range(n):
    # ODE solution (solid line)
    plt.plot(solution.t, solution.y[i], label=f"ODE: q{i+1}")
    # MuJoCo simulation (dashed line)
    plt.plot(t_eval_mujoco, qpos[:, i], '--', label=f"MuJoCo: q{i+1}")

plt.title("Generalized Coordinates Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Generalized Coordinates (q)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
