###############################################################################
# This script derives the ODE for the drone with spring-pendulum
#   1) Builds the Lagrangian for the 2D quadrotor + spring pendulum system.
#   2) Derives the implicit EOM: d/dt(∂L/∂q_dot) - ∂L/∂q = Q.
#   3) Symbolically solves for q_ddot => obtains an explicit ODE form.
#   4) Assembles the first-order system x_dot = f(x, u).
#   5) Post-processes the symbolic result to reference x[0], x[1], ...
#      instead of "y(t)" or "Derivative(y(t), t)". Also replaces sin->np.sin, etc.
#   6) Saves a Python file "eom_2d_quadro_springpend_explicit.py" containing
#      a function eom_2d_quadro_springpend_explicit(x, u, params) -> x_dot
###############################################################################

import sympy
import os

# determine whether to save as casadi or numpy
casadi = True
# determine whether to add damper or not
use_damper = True

###############################################################################
# A) Directory setup
###############################################################################
save_dir = os.path.dirname(__file__)
os.makedirs(save_dir, exist_ok=True)

###############################################################################
# B) Define symbols and dynamic variables
###############################################################################
t = sympy.Symbol('t', real=True)

# Parameters
M   = sympy.Symbol('M',     real=True)  # Drone mass
m   = sympy.Symbol('m',     real=True)  # Pendulum mass
Ixx = sympy.Symbol('Ixx',   real=True)  # Drone inertia about CoM
g   = sympy.Symbol('g',     real=True)  
k   = sympy.Symbol('k',     real=True)  # Spring constant
l0  = sympy.Symbol('l0',    real=True)  # Spring rest length
c  = sympy.Symbol('c',     real=True)  # rotor thrust coefficient
L_rot   = sympy.Symbol('L_rot',     real=True)  # half-distance between rotors
d   = sympy.Symbol('d',     real=True)  # Spring-damper coefficient (N*s/m)

# External forces
f_ext_dy = sympy.Symbol('f_ext_dy', real=True)
f_ext_dz = sympy.Symbol('f_ext_dz', real=True)
f_ext_my = sympy.Symbol('f_ext_my', real=True)
f_ext_mz = sympy.Symbol('f_ext_mz', real=True)

# Controls (u1 = dw1/dt, u2 = dw2/dt)
u1 = sympy.Symbol('u1', real=True)
u2 = sympy.Symbol('u2', real=True)

# Generalized coords & rotor speeds as functions of t
y     = sympy.Function('y')(t)
z     = sympy.Function('z')(t)
phi   = sympy.Function('phi')(t)
r     = sympy.Function('r')(t)
theta = sympy.Function('theta')(t)
w1    = sympy.Function('w1')(t)
w2    = sympy.Function('w2')(t)

# Derivatives
y_dot     = y.diff(t)
z_dot     = z.diff(t)
phi_dot   = phi.diff(t)
r_dot     = r.diff(t)
theta_dot = theta.diff(t)
w1_dot    = w1.diff(t)
w2_dot    = w2.diff(t)

###############################################################################
# C) Build Kinetic & Potential Energy
###############################################################################
# Drone's KE
T_drone = 0.5*M*(y_dot**2 + z_dot**2) + 0.5*Ixx*(phi_dot**2)

# Pendulum bob coords
y_m = y + r*sympy.sin(theta)
z_m = z - r*sympy.cos(theta)
y_m_dot = y_m.diff(t)
z_m_dot = z_m.diff(t)

# Pendulum bob's KE
T_pend = 0.5*m*(y_m_dot**2 + z_m_dot**2)

T_total = T_drone + T_pend

# Potential energies
U_drone  = M*g*z
U_mass   = m*g*z_m
U_spring = 0.5*k*(r - l0)**2

U_total = U_drone + U_mass + U_spring

# Lagrangian
L = T_total - U_total

###############################################################################
# D) Non-conservative forces Q
###############################################################################
# Thrust, rotor torque
f_thrust   = c*(w1 + w2)
tau_thrust = c*L_rot*(w2 - w1)

# --- Qy: force in the y-direction on the drone + mass
Qy_drone = f_ext_dy + f_thrust*(-sympy.sin(phi))   # thrust + ext on drone
Qy_mass  = f_ext_my                                # from mass's external force 
#    partial derivative wrt y => mass moves 1:1 with y => dot product = f_ext_my, y_mass = y + rsin(phi)!
Qy_total = Qy_drone + Qy_mass

# --- Qz: force in the z-direction on the drone + mass
Qz_drone = f_ext_dz + f_thrust*sympy.cos(phi)      # thrust + ext on drone
Qz_mass  = f_ext_mz
Qz_total = Qz_drone + Qz_mass

# --- Qphi: torque about drone's CoM from thrust + external
#    The external force at CoM (f_ext_d) has zero moment about CoM (no arm),
#    so Qphi from drone external force is 0 unless an offset is given.
#    We do have rotor torque:
Qphi_total = tau_thrust

#    For mass external force, the pivot is the drone’s CoM, so no offset => 0.

# --- Qr: partial derivative wrt 'r'
#    The external force on the mass is f_ext_m = (f_ext_my, f_ext_mz).
#    Pendulum bob position (y_m, z_m) depends on r => ∂(y_m,z_m)/∂r
#          = ( sin(theta), -cos(theta) )
Qr_mass = f_ext_my*sympy.sin(theta) + f_ext_mz*(-sympy.cos(theta))
#    Drone external force and thrust do not depend on r. 
Qr_total = Qr_mass
if use_damper:
    Qr_total -= d*r_dot

# --- Qtheta: partial derivative wrt 'theta'
#     (y_m, z_m) = (y + r sin(theta), z - r cos(theta))
#     ∂(y_m,z_m)/∂theta = (r cos(theta), r sin(theta))
Qtheta_mass = f_ext_my*(r*sympy.cos(theta)) + f_ext_mz*(r*sympy.sin(theta))
Qtheta_total = Qtheta_mass

coords     = [y, z, phi, r, theta]
coords_dot = [y_dot, z_dot, phi_dot, r_dot, theta_dot]
Q_list     = [Qy_total, Qz_total, Qphi_total, Qr_total, Qtheta_total]

###############################################################################
# E) Form the implicit EOM: d/dt(∂L/∂q_dot) - ∂L/∂q = Q
###############################################################################
dL_dq     = [L.diff(q)     for q     in coords]
dL_dq_dot = [L.diff(qd)    for qd    in coords_dot]

eom_list = []
for i in range(len(coords)):
    eq_i = dL_dq_dot[i].diff(t) - dL_dq[i] - Q_list[i]
    eom_list.append(eq_i)

# We'll solve these for the second derivatives (q_ddot).
q_ddot_syms = [sympy.Symbol(f'q_ddot_{i}', real=True) for i in range(5)]
subs_ddot = {coords_dot[i].diff(t): q_ddot_syms[i] for i in range(5)}
eom_subs = [eq.subs(subs_ddot) for eq in eom_list]

sol = sympy.solve(eom_subs, q_ddot_syms, dict=True)
if not sol:
    raise RuntimeError("No solution for q_ddot found.")
if len(sol) > 1:
    raise RuntimeError("Multiple solutions for q_ddot? That's unexpected here.")
sol_ddot = sol[0]  # e.g. {q_ddot_0: expr_for_ddot_y, q_ddot_1: ...}

###############################################################################
# F) Construct the 12D state and its derivative
#    x = [ y, z, phi, r, theta, y_dot, z_dot, phi_dot, r_dot, theta_dot, w1, w2 ]
#    x_dot = [ y_dot, z_dot, phi_dot, r_dot, theta_dot, ddot_y, ddot_z, ddot_phi,
#              ddot_r, ddot_theta, w1_dot, w2_dot ]
###############################################################################

# Build symbolic expressions for each x_dot[i].
x_dot_sym = []

# (1) y_dot, z_dot, phi_dot, r_dot, theta_dot
x_dot_sym += [y_dot, z_dot, phi_dot, r_dot, theta_dot]

# (2) ddot_y, ddot_z, ddot_phi, ddot_r, ddot_theta
for i in range(5):
    x_dot_sym.append(sol_ddot[q_ddot_syms[i]])

# (3) w1_dot, w2_dot = u1, u2
x_dot_sym.append(u1)
x_dot_sym.append(u2)

###############################################################################
# G) Post-processing: Replace references to y(t), Derivative(y(t), t), etc.
#    with x[0], x[5], etc.  Then replace trig calls with np.xxx
###############################################################################
#
# We'll do two passes: (A) Replacements for 'y(t)', 'Derivative(y(t), t)', ...
#                     (B) Then replacements for 'Derivative(x[0], t)' if they appear
#                        (some Sympy expansions can produce them).
#
# Alternatively, we can do everything in a single dictionary if we carefully
# include both 'y(t)' -> 'x[0]' and 'Derivative(x[0], t)' -> 'x[5]'.
###############################################################################

repl_pass_1_casadi = {
    'y(t)':       'x[0]',
    'z(t)':       'x[1]',
    'phi(t)':     'x[2]',
    'r(t)':       'x[3]',
    'theta(t)':   'x[4]',
    
    'Derivative(y(t), t)':       'x[5]',
    'Derivative(z(t), t)':       'x[6]',
    'Derivative(phi(t), t)':     'x[7]',
    'Derivative(r(t), t)':       'x[8]',
    'Derivative(theta(t), t)':   'x[9]',

    'w1(t)': 'x[10]',
    'w2(t)': 'x[11]',

    'u1': 'u[0]',
    'u2': 'u[1]',

    'sin': 'ca.sin',
    'cos': 'ca.cos',
    'tan': 'ca.tan',
    'asin': 'ca.arcsin',
    'acos': 'ca.arccos',
    'atan': 'ca.arctan',
    'exp': 'ca.exp',
}

repl_pass_1_numpy = {
    'y(t)':       'x[0]',
    'z(t)':       'x[1]',
    'phi(t)':     'x[2]',
    'r(t)':       'x[3]',
    'theta(t)':   'x[4]',
    
    'Derivative(y(t), t)':       'x[5]',
    'Derivative(z(t), t)':       'x[6]',
    'Derivative(phi(t), t)':     'x[7]',
    'Derivative(r(t), t)':       'x[8]',
    'Derivative(theta(t), t)':   'x[9]',

    'w1(t)': 'x[10]',
    'w2(t)': 'x[11]',

    'u1': 'u[0]',
    'u2': 'u[1]',

    'sin': 'np.sin',
    'cos': 'np.cos',
    'tan': 'np.arcsin',
    'acos': 'np.arccos',
    'atan': 'np.arctan',
    'exp': 'np.exp',
}

# In case the expression contains "Derivative(x[0], t)" after the first pass,
# we do a second pass:
repl_pass_2 = {
    'Derivative(x[0], t)': 'x[5]',
    'Derivative(x[1], t)': 'x[6]',
    'Derivative(x[2], t)': 'x[7]',
    'Derivative(x[3], t)': 'x[8]',
    'Derivative(x[4], t)': 'x[9]',
}

def expr_to_python_str(expr):
    """ Convert a Sympy expression into a pythonic string with x[...] references. """
    # Optional: simplify, can be large
    expr_simpl = sympy.simplify(expr)
    s = str(expr_simpl)
    # 1) pass one
    if casadi:
        for old, new in repl_pass_1_casadi.items():
            s = s.replace(old, new)
    else:
        for old, new in repl_pass_1_numpy.items():
            s = s.replace(old, new)
    # 2) pass two
    for old, new in repl_pass_2.items():
        s = s.replace(old, new)
    return s

x_dot_str_list = [expr_to_python_str(e) for e in x_dot_sym]

###############################################################################
# H) Write out the final function eom_2d_quadro_springpend_explicit.py
###############################################################################

# Decide naming and parameter docstring based on damper toggle
tag = "springdamppend" if use_damper else "springpend"
func_name = f"eom_2d_quadro_{tag}_explicit"
param_list_str = (
    "[M, m, Ixx, g, k, l0, c, L_rot, f_ext_dy, f_ext_dz, f_ext_my, f_ext_mz, d]"
    if use_damper else
    "[M, m, Ixx, g, k, l0, c, L_rot, f_ext_dy, f_ext_dz, f_ext_my, f_ext_mz]"
)

if casadi:
    filename = os.path.join(save_dir, f"{func_name}_casadi.py")
    with open(filename, 'w') as f:
        f.write("import casadi as ca\n\n")
        f.write(f"def {func_name}(x, u, params):\n")
        f.write("    # x = [ y, z, phi, r, theta, y_dot, z_dot, phi_dot, r_dot, theta_dot, w1, w2 ]\n")
        f.write("    # u = [u1, u2] = [dw1/dt, dw2/dt]\n")
        f.write(f"    # params = {param_list_str}\n\n")
        f.write("    params_list = ca.vertsplit(params)\n")
        f.write("    M = params_list[0]\n")
        f.write("    m = params_list[1]\n")
        f.write("    Ixx = params_list[2]\n")
        f.write("    g = params_list[3]\n")
        f.write("    k = params_list[4]\n")
        f.write("    l0 = params_list[5]\n")
        f.write("    c = params_list[6]\n")
        f.write("    L_rot = params_list[7]\n")
        f.write("    f_ext_dy = params_list[8]\n")
        f.write("    f_ext_dz = params_list[9]\n")
        f.write("    f_ext_my = params_list[10]\n")
        f.write("    f_ext_mz = params_list[11]\n")
        if use_damper:
            f.write("    d = params_list[12]\n\n")
        f.write("    x_dot = ca.MX.zeros(12)\n\n")
        for i, rhs_str in enumerate(x_dot_str_list):
            f.write(f"    x_dot[{i}] = {rhs_str}\n")
        f.write("    return x_dot\n")

    print(f"Explicit ODE code saved to: {filename}")
else:
    filename = os.path.join(save_dir, f"{func_name}_numpy.py")
    with open(filename, 'w') as f:
        f.write("import numpy as np\n\n")
        f.write(f"def {func_name}(x, u, params):\n")
        f.write("    # x = [ y, z, phi, r, theta, y_dot, z_dot, phi_dot, r_dot, theta_dot, w1, w2 ]\n")
        f.write("    # u = [u1, u2] = [dw1/dt, dw2/dt]\n")
        f.write(f"    # params = {param_list_str}\n\n")
        if not use_damper:
            f.write("    M, m, Ixx, g, k, l0, c, L_rot, f_ext_dy, f_ext_dz, f_ext_my, f_ext_mz = params\n\n")
        else:
            f.write("    M, m, Ixx, g, k, l0, c, L_rot, f_ext_dy, f_ext_dz, f_ext_my, f_ext_mz, d = params\n\n")
        f.write("    x_dot = np.zeros(12)\n\n")
        for i, rhs_str in enumerate(x_dot_str_list):
            f.write(f"    x_dot[{i}] = {rhs_str}\n")
        f.write("    return x_dot\n")

    print(f"Explicit ODE code saved to: {filename}")
