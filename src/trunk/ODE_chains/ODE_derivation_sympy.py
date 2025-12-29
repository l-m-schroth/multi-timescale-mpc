"""
Derives the trunk ODE using Lagrangian dynamics and sympy
"""

from sympy import symbols, Function, diff, Matrix, sin, cos, simplify, trigsimp, Rational
import os

# --- Directory setup ---
save_dir = os.path.join(os.path.dirname(__file__))
os.makedirs(save_dir, exist_ok=True)

# --- Simplification switch ---
simplify_flag = False  # Set to True to enable simplification, False to skip it

# --- Definitions and parameters ---
n = 16 # Number of links in the chain
t, g, m, l, Iz, c, d, gear = symbols('t g m l Iz c d gear')
q_list = [Function(f'q{i}')(t) for i in range(1, n + 1)]  # Generalized coordinates
q_dot_list = [diff(q, t) for q in q_list]  # Generalized velocities
q = Matrix(q_list)
q_dot = Matrix(q_dot_list)

# --- COM positions ---
x_list = []
y_list = []

for i in range(n):
    x_pos = 0
    y_pos = 0
    q_tot = 0
    for j in range(i):
        q_tot += q[j]
        x_pos += 2 * l * sin(q_tot)
        y_pos -= 2 * l * cos(q_tot)
    x_pos += l * sin(q_tot + q[i])
    y_pos -= l * cos(q_tot + q[i])
    x_list.append(x_pos)
    y_list.append(y_pos)

# --- Kinetic energy ---
v_tot_sq = 0
for i in range(n):
    x_dot = diff(x_list[i], t)
    y_dot = diff(y_list[i], t)
    v_tot_sq += x_dot**2 + y_dot**2
T_lin = Rational(1, 2) * m * v_tot_sq

omega_tot_sq = sum(sum(q_dot[j] for j in range(i + 1))**2 for i in range(n))
T_rot = Rational(1, 2) * Iz * omega_tot_sq

T = T_lin + T_rot

# --- Potential energy ---
U_springs = Rational(1, 2) * c * sum(q[i]**2 for i in range(n))
U_grav = m * g * sum(y_list)
U = U_grav + U_springs

# --- Generalized torque ---
tau_damping = -d * q_dot
tau_controls = gear*Matrix([symbols(f'u[{i}]') for i in range(n)])
tau = tau_damping + tau_controls

# --- Lagrangian and equations of motion ---
L = T - U
dL_dq = Matrix([diff(L, q[i]) for i in range(n)])
dL_dq_dot = Matrix([diff(L, q_dot[i]) for i in range(n)])
ddt_dL_dq_dot = Matrix([diff(dL_dq_dot[i], t) for i in range(n)]).subs(
    {diff(q[i], t): q_dot[i] for i in range(n)}
)

# Equations of motion in implicit form (all terms on one side, equals zero)
eom_implicit = ddt_dL_dq_dot - dL_dq - tau

# Apply simplifications if the flag is set
if simplify_flag:
    eom_result = Matrix([trigsimp(simplify(eom_implicit[i])) for i in range(n)])
else:
    eom_result = eom_implicit

# --- Convert to numerical functions ---
# Replace second derivatives with numerical variables
q_ddot_list = [symbols(f'q_ddot[{i}]') for i in range(n)]
eom_substituted = eom_result.subs(
    {diff(q[i], t, t): q_ddot_list[i] for i in range(n)}
).subs(
    {diff(q[i], t): symbols(f'q_dot[{i}]') for i in range(n)}
).subs(
    {q[i]: symbols(f'q[{i}]') for i in range(n)}
)

# Replace trigonometric functions with NumPy equivalents
eom_numpy = [str(eq).replace('sin', 'np.sin').replace('cos', 'np.cos') for eq in eom_substituted]

# Print the equations (substituted form with NumPy functions)
print("Equations of Motion (Implicit Form with q, q_dot, q_ddot):")
for i, eq in enumerate(eom_numpy):
    print(f"Equation {i+1}: {eq} = 0")

# Save the substituted equations to a Python file
file_path = os.path.join(save_dir, f'eom_system_{n}.py')
with open(file_path, 'w') as f:
    f.write("import numpy as np\n\n")
    f.write(f"def eom_{n}(q, q_dot, q_ddot, u, params):\n")
    f.write("    g, m, l, Iz, c, d, gear = params\n")
    f.write("    eqs = np.array([\n")
    for eq in eom_numpy:
        f.write(f"        {eq},\n")
    f.write("    ])\n")
    f.write("    return eqs\n")

print(f"ODE function saved in directory: {save_dir}")

# Save the substituted equations to a CasADi-compatible Python file
casadi_file_path = os.path.join(save_dir, f'eom_system_{n}_casadi.py')
with open(casadi_file_path, 'w') as f:
    f.write("import casadi as ca\n\n")
    f.write(f"def eom_{n}(q, q_dot, q_ddot, u, params):\n")
    f.write("    g, m, l, Iz, c, d, gear = params\n")
    f.write("    eqs = ca.vertcat(\n")
    for eq in eom_numpy:
        casadi_eq = eq.replace('np.', 'ca.')  # Replace NumPy functions with CasADi
        f.write(f"        {casadi_eq},\n")
    f.write("    )\n")
    f.write("    return eqs\n")

print(f"CasADi-compatible ODE function saved in directory: {save_dir}")
