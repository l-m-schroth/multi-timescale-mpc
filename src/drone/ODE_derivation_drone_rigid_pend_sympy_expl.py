###############################################################################
# This script derives the ODE for the drone with rigid pendulum
#   1) Builds the Lagrangian for the 2D quadrotor + *fixed-length* pendulum.
#   2) Derives the implicit EOM: d/dt(∂L/∂q_dot) - ∂L/∂q = Q.
#   3) Symbolically solves for q_ddot => obtains an explicit ODE form.
#   4) Assembles the first-order system x_dot = f(x, u).
#   5) Post-processes the symbolic result to reference x[0], x[1], ...
#      instead of "y(t)" or "Derivative(y(t), t)". Also replaces sin->np.sin, etc.
#   6) Saves a Python file "eom_2d_quadro_pend_explicit.py" containing
#      a function eom_2d_quadro_pend_explicit(x, u, params) -> x_dot
###############################################################################

import sympy
import os

# Toggle whether to generate a CasADi version or a NumPy version
casadi = True

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
M     = sympy.Symbol('M',     real=True)  # Drone mass
m     = sympy.Symbol('m',     real=True)  # Pendulum mass
Ixx   = sympy.Symbol('Ixx',   real=True)  # Drone inertia about CoM
g     = sympy.Symbol('g',     real=True)
l0    = sympy.Symbol('l0',    real=True)  # Pendulum length (fixed)
c     = sympy.Symbol('c',     real=True)  # rotor thrust coefficient
L_rot = sympy.Symbol('L_rot', real=True)  # half-distance between rotors

# External forces
f_ext_dy = sympy.Symbol('f_ext_dy', real=True)  # external force on drone in y
f_ext_dz = sympy.Symbol('f_ext_dz', real=True)  # external force on drone in z
f_ext_my = sympy.Symbol('f_ext_my', real=True)  # external force on pend mass in y
f_ext_mz = sympy.Symbol('f_ext_mz', real=True)  # external force on pend mass in z

# Controls (u1 = dw1/dt, u2 = dw2/dt)
u1 = sympy.Symbol('u1', real=True)
u2 = sympy.Symbol('u2', real=True)

# Generalized coords & rotor speeds as functions of t
y     = sympy.Function('y')(t)
z     = sympy.Function('z')(t)
phi   = sympy.Function('phi')(t)
theta = sympy.Function('theta')(t)
w1    = sympy.Function('w1')(t)
w2    = sympy.Function('w2')(t)

# Derivatives
y_dot     = y.diff(t)
z_dot     = z.diff(t)
phi_dot   = phi.diff(t)
theta_dot = theta.diff(t)
w1_dot    = w1.diff(t)
w2_dot    = w2.diff(t)

###############################################################################
# C) Build Kinetic & Potential Energy
###############################################################################
# Drone's KE
T_drone = 0.5*M*(y_dot**2 + z_dot**2) + 0.5*Ixx*(phi_dot**2)

# Pendulum bob coordinates:
#   y_m = y + l0 * sin(theta)
#   z_m = z - l0 * cos(theta)
y_m = y + l0*sympy.sin(theta)
z_m = z - l0*sympy.cos(theta)

y_m_dot = y_m.diff(t)
z_m_dot = z_m.diff(t)

# Pendulum bob's KE
T_pend = 0.5*m*(y_m_dot**2 + z_m_dot**2)

T_total = T_drone + T_pend

# Potential energies
U_drone = M*g*z
U_mass  = m*g*z_m  # no spring term; length is fixed

U_total = U_drone + U_mass

# Lagrangian
L = T_total - U_total

###############################################################################
# D) Non-conservative forces Q
###############################################################################
# Thrust, rotor torque
f_thrust   = c*(w1 + w2)
tau_thrust = c*L_rot*(w2 - w1)

# Qy: total generalized force in y
#   Drone receives external + thrust in −sin(phi),
#   Pendulum bob has external forces f_ext_my.
Qy_drone = f_ext_dy + f_thrust*(-sympy.sin(phi))
Qy_mass  = f_ext_my
Qy_total = Qy_drone + Qy_mass

# Qz: total generalized force in z
Qz_drone = f_ext_dz + f_thrust*sympy.cos(phi)
Qz_mass  = f_ext_mz
Qz_total = Qz_drone + Qz_mass

# Qphi: torque about drone's CoM
Qphi_total = tau_thrust
# (External force at CoM has no moment about CoM; pend mass external has pivot at drone, so zero offset.)

# Qtheta: partial derivative wrt theta.
#   Pendulum bob position: (y_m, z_m) = (y + l0 sin(theta), z - l0 cos(theta))
#   => ∂(y_m,z_m)/∂theta = (l0 cos(theta), l0 sin(theta))
#   => Qtheta = f_ext_my*(l0 cos(theta)) + f_ext_mz*(l0 sin(theta))
Qtheta_mass = f_ext_my*(l0*sympy.cos(theta)) + f_ext_mz*(l0*sympy.sin(theta))
Qtheta_total = Qtheta_mass

coords     = [y, z, phi, theta]
coords_dot = [y_dot, z_dot, phi_dot, theta_dot]
Q_list     = [Qy_total, Qz_total, Qphi_total, Qtheta_total]

###############################################################################
# E) Form the implicit EOM: d/dt(∂L/∂q_dot) - ∂L/∂q = Q
###############################################################################
dL_dq     = [L.diff(q) for q in coords]
dL_dq_dot = [L.diff(qd) for qd in coords_dot]

eom_list = []
for i in range(len(coords)):
    eq_i = dL_dq_dot[i].diff(t) - dL_dq[i] - Q_list[i]
    eom_list.append(eq_i)

# We'll solve these for the second derivatives (q_ddot).
q_ddot_syms = [sympy.Symbol(f'q_ddot_{i}', real=True) for i in range(len(coords))]
subs_ddot   = {coords_dot[i].diff(t): q_ddot_syms[i] for i in range(len(coords))}
eom_subs    = [eq.subs(subs_ddot) for eq in eom_list]

sol = sympy.solve(eom_subs, q_ddot_syms, dict=True)
if not sol:
    raise RuntimeError("No solution for q_ddot found.")
if len(sol) > 1:
    raise RuntimeError("Multiple solutions for q_ddot? That's unexpected here.")
sol_ddot = sol[0]

###############################################################################
# F) Construct the 10D state and its derivative
#    x = [ y, z, phi, theta, y_dot, z_dot, phi_dot, theta_dot, w1, w2 ]
#    x_dot = [ y_dot, z_dot, phi_dot, theta_dot, ddot_y, ddot_z, ddot_phi, ddot_theta, w1_dot, w2_dot ]
###############################################################################

x_dot_sym = []

# 1) y_dot, z_dot, phi_dot, theta_dot
x_dot_sym += [y_dot, z_dot, phi_dot, theta_dot]

# 2) ddot_y, ddot_z, ddot_phi, ddot_theta
for i in range(len(coords)):
    x_dot_sym.append(sol_ddot[q_ddot_syms[i]])

# 3) w1_dot, w2_dot = u1, u2
x_dot_sym.append(u1)
x_dot_sym.append(u2)

###############################################################################
# G) Post-processing: Replace references to y(t), Derivative(y(t), t), etc.
#    with x[0], x[4], etc. Then replace trig calls with np.xxx or ca.xxx
###############################################################################

repl_pass_1_casadi = {
    'y(t)':       'x[0]',
    'z(t)':       'x[1]',
    'phi(t)':     'x[2]',
    'theta(t)':   'x[3]',
    
    'Derivative(y(t), t)':       'x[4]',
    'Derivative(z(t), t)':       'x[5]',
    'Derivative(phi(t), t)':     'x[6]',
    'Derivative(theta(t), t)':   'x[7]',

    'w1(t)': 'x[8]',
    'w2(t)': 'x[9]',

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
    'theta(t)':   'x[3]',

    'Derivative(y(t), t)':       'x[4]',
    'Derivative(z(t), t)':       'x[5]',
    'Derivative(phi(t), t)':     'x[6]',
    'Derivative(theta(t), t)':   'x[7]',

    'w1(t)': 'x[8]',
    'w2(t)': 'x[9]',

    'u1': 'u[0]',
    'u2': 'u[1]',

    'sin': 'np.sin',
    'cos': 'np.cos',
    'tan': 'np.tan',
    'asin': 'np.arcsin',
    'acos': 'np.arccos',
    'atan': 'np.arctan',
    'exp': 'np.exp',
}

# Second pass in case "Derivative(x[...], t)" appears
repl_pass_2 = {
    'Derivative(x[0], t)': 'x[4]',
    'Derivative(x[1], t)': 'x[5]',
    'Derivative(x[2], t)': 'x[6]',
    'Derivative(x[3], t)': 'x[7]',
}

def expr_to_python_str(expr):
    """Convert a Sympy expression into a pythonic string with x[...] references."""
    expr_simpl = sympy.simplify(expr)
    s = str(expr_simpl)
    # pass one
    if casadi:
        for old, new in repl_pass_1_casadi.items():
            s = s.replace(old, new)
    else:
        for old, new in repl_pass_1_numpy.items():
            s = s.replace(old, new)
    # pass two
    for old, new in repl_pass_2.items():
        s = s.replace(old, new)
    return s

x_dot_str_list = [expr_to_python_str(e) for e in x_dot_sym]

###############################################################################
# H) Write out the final function eom_2d_quadro_pend_explicit.py
###############################################################################
if casadi:
    filename = os.path.join(save_dir, "eom_2d_quadro_pend_explicit_casadi.py")
    with open(filename, 'w') as f:
        f.write("import casadi as ca\n\n")
        f.write("def eom_2d_quadro_pend_explicit(x, u, params):\n")
        f.write("    # x = [ y, z, phi, theta, y_dot, z_dot, phi_dot, theta_dot, w1, w2 ]\n")
        f.write("    # u = [u1, u2] = [dw1/dt, dw2/dt]\n")
        f.write("    # params = [M, m, Ixx, g, l0, c, L_rot, f_ext_dy, f_ext_dz, f_ext_my, f_ext_mz]\n\n")
        f.write("    params_list = ca.vertsplit(params)\n")
        f.write("    M = params_list[0]\n")
        f.write("    m = params_list[1]\n")
        f.write("    Ixx = params_list[2]\n")
        f.write("    g = params_list[3]\n")
        f.write("    l0 = params_list[4]\n")
        f.write("    c = params_list[5]\n")
        f.write("    L_rot = params_list[6]\n")
        f.write("    f_ext_dy = params_list[7]\n")
        f.write("    f_ext_dz = params_list[8]\n")
        f.write("    f_ext_my = params_list[9]\n")
        f.write("    f_ext_mz = params_list[10]\n\n")
        f.write("    x_dot = ca.MX.zeros(10)\n\n")
        for i, rhs_str in enumerate(x_dot_str_list):
            f.write(f"    x_dot[{i}] = {rhs_str}\n")
        f.write("    return x_dot\n")
    print(f"Explicit ODE code saved to: {filename}")

else:
    filename = os.path.join(save_dir, "eom_2d_quadro_pend_explicit_numpy.py")
    with open(filename, 'w') as f:
        f.write("import numpy as np\n\n")
        f.write("def eom_2d_quadro_pend_explicit(x, u, params):\n")
        f.write("    # x = [ y, z, phi, theta, y_dot, z_dot, phi_dot, theta_dot, w1, w2 ]\n")
        f.write("    # u = [u1, u2] = [dw1/dt, dw2/dt]\n")
        f.write("    # params = [M, m, Ixx, g, l0, c, L_rot, f_ext_dy, f_ext_dz, f_ext_my, f_ext_mz]\n\n")
        f.write("    M, m, Ixx, g, l0, c, L_rot, f_ext_dy, f_ext_dz, f_ext_my, f_ext_mz = params\n\n")
        f.write("    x_dot = np.zeros(10)\n\n")
        for i, rhs_str in enumerate(x_dot_str_list):
            f.write(f"    x_dot[{i}] = {rhs_str}\n")
        f.write("    return x_dot\n")
    print(f"Explicit ODE code saved to: {filename}")
