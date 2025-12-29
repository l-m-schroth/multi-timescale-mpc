# This file can be used to sanity check the derived ODE with spring-pendulum

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------
# 1) Import your ODE function from the generated file:
#    If this file is in the same directory, just do:
#        from eom_2d_quadro_springpend_explicit_numpy import eom_2d_quadro_springpend_explicit
#    Adjust the import path if needed.
# ---------------------------------------------------------------------
from drone.eom_2d_quadro_springpend_explicit_numpy import eom_2d_quadro_springpend_explicit

# ---------------------------------------------------------------------
# 2) Define a helper function to integrate the ODE given an initial
#    condition x0, time range, parameters, and a control law.
# ---------------------------------------------------------------------
def simulate_system(x0, t_span, params, control_law, num_steps=300):
    """
    x0:       array-like, shape (12,): initial state
    t_span:   (t_start, t_end), time range for integration
    params:   array-like of system parameters
    control_law: function(t, x) -> u (2D vector for rotor spin accelerations)
    num_steps: how many time points to sample for plotting

    returns:
      t_eval:  time array (shape (num_steps,))
      x_sol:   array of shape (num_steps, 12) with the solution
    """

    def ode_func(t, x):
        # Evaluate the control inputs
        u = control_law(t, x)
        return eom_2d_quadro_springpend_explicit(x, u, params)

    # Create a time grid (for stable sampling & animation, if desired)
    t_eval = np.linspace(t_span[0], t_span[1], num_steps)

    # Solve IVP
    sol = solve_ivp(ode_func, t_span, x0, t_eval=t_eval, rtol=1e-8, atol=1e-8)

    # sol.y is shape (states, len(t_eval)), transpose it to (len(t_eval), states)
    x_sol = sol.y.T
    return sol.t, x_sol

# ---------------------------------------------------------------------
# 3) A helper to plot the first five states: y, z, theta, r, phi
# ---------------------------------------------------------------------
def plot_states(t, x_sol, scenario_name):
    """
    Plots the first five states (y, z, theta, r, phi) vs time
    in a single figure with subplots.
    """
    # x = [ y, z, phi, r, theta, y_dot, z_dot, phi_dot, r_dot, theta_dot, w1, w2 ]
    # Index mapping:
    #   y -> x_sol[:, 0]
    #   z -> x_sol[:, 1]
    #   phi -> x_sol[:, 2]
    #   r -> x_sol[:, 3]
    #   theta -> x_sol[:, 4]

    fig = plt.figure()
    fig.suptitle(f"Scenario: {scenario_name}")

    labels = ["y", "z", "phi", "r", "theta"]
    indices = [0, 1, 2, 3, 4]  # note that we want [y, z, theta, r, phi]

    for i, (lbl, idx) in enumerate(zip(labels, indices), start=1):
        ax = fig.add_subplot(5, 1, i)
        ax.plot(t, x_sol[:, idx])  # let matplotlib pick default colors
        ax.set_ylabel(lbl)
        if i < 5:
            ax.set_xticklabels([])  # hide x ticks except for the last subplot

    ax.set_xlabel("time (s)")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# 4) Main demonstration of the 3 scenarios
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # -----------------------------------------------------------------
    # Define system parameters. Format:
    #   params = [M, m, Ixx, g, k, l0, c, L_rot, f_ext_dy, f_ext_dz, f_ext_my, f_ext_mz]
    # We'll choose a fairly stiff spring (k=50) and a light pendulum mass (m=0.05),
    # so the pendulum doesn't throw the drone around too much.
    # -----------------------------------------------------------------
    params = [
        1.0,    # M (drone mass)
        0.05,   # m (pendulum mass) -- quite light
        0.01,   # Ixx (drone inertia)
        9.81,   # g
        50.0,   # k (spring constant) -> quite stiff
        0.5,    # l0 (rest length)
        1.0,    # c (thrust coefficient)
        0.2,    # L_rot (half-distance between rotors)
        0.0,    # f_ext_dy
        0.0,    # f_ext_dz
        0.0,    # f_ext_my
        0.0     # f_ext_mz
    ]

    # For convenience in these examples, we'll define a simple control law
    # that sets rotor spin *accelerations* to zero => rotor speeds remain constant.
    def zero_control_law(t, x):
        return np.array([0.0, 0.0])

    # We'll define a helper to find the rotor spin rates that produce net thrust
    # of (M+m)*g in equilibrium:
    M = params[0]
    m = params[1]
    c_ = params[6]
    net_weight = (M + m) * params[3]  # (M+m)*g
    # we want f = c*(w1 + w2) = net_weight => w1 + w2 = net_weight / c
    # for a balanced hover, let w1 = w2:
    hover_spin = 0.5 * net_weight / c_  # w1 = w2

    # -----------------------------------------------------------------
    # Scenario 1: Everything at rest NOTE: still oscillating mass due to gravity, which also cause z to minimally oscillate
    #  - The drone and mass are both at y=0, z=0, r=l0, phi=0, theta=0.
    #  - The rotors are spinning so that net thrust = (M+m)*g.
    #  - The control input is zero => spin rates don't change.
    #  - We expect no motion if this is indeed an equilibrium.
    # -----------------------------------------------------------------
    x0_scenario1 = np.array([
        0.0,          # y
        0.0,          # z
        0.0,          # phi
        0.5,          # r = l0
        0.0,          # theta
        0.0,          # y_dot
        0.0,          # z_dot
        0.0,          # phi_dot
        0.0,          # r_dot
        0.0,          # theta_dot
        hover_spin,   # w1
        hover_spin    # w2
    ])

    t_span = (0, 5)
    t_s1, x_s1 = simulate_system(
        x0_scenario1, t_span, params, zero_control_law
    )
    plot_states(t_s1, x_s1, scenario_name="1) Perfect Hover (Check equilibrium)")

    # -----------------------------------------------------------------
    # Scenario 2: Same as scenario 1, but we give the rotors *slightly*
    # higher constant spin so there's a net positive thrust and it climbs.
    # -----------------------------------------------------------------
    # Let's add a small offset to hover_spin:
    w_up = hover_spin + 0.1
    x0_scenario2 = x0_scenario1.copy()
    x0_scenario2[10] = w_up  # w1
    x0_scenario2[11] = w_up  # w2

    t_s2, x_s2 = simulate_system(
        x0_scenario2, (0, 5), params, zero_control_law
    )
    plot_states(t_s2, x_s2, scenario_name="2) Slightly More Thrust => Slow Climb")

    # -----------------------------------------------------------------
    # Scenario 3: Drone in hover, but the pendulum is not at resting
    # position. We'll offset r slightly above l0, or give theta an offset,
    # to watch it oscillate. Meanwhile, the drone is large enough and the
    # pendulum is light enough that the drone won't move much.
    # -----------------------------------------------------------------
    x0_scenario3 = x0_scenario1.copy()
    x0_scenario3[3] = 0.6      # r = 0.6 instead of 0.5 => slight stretch
    x0_scenario3[4] = 0.2      # theta = 0.2 rad => small angular displacement

    t_s3, x_s3 = simulate_system(
        x0_scenario3, (0, 5), params, zero_control_law
    )
    plot_states(t_s3, x_s3, scenario_name="3) Pendulum Oscillation at Hover")

    # -----------------------------------------------------------------
    # Scenario 4: Side wind forcing the drone in +y direction.
    # We set params[8] = f_ext_dy to some positive value (like 0.2).
    # We'll keep the rest the same, including the rotor speeds at hover.
    #
    # Expected outcome: The drone is no longer able to stay perfectly
    # at y=0 because the horizontal external force is unopposed (our
    # control is zero rotor spin acceleration, so the drone's net thrust
    # remains vertical). The drone will drift in +y, and the pendulum may
    # swing slightly. But z should remain close to 0 if thrust = weight.
    # -----------------------------------------------------------------
    params_s4 = params.copy()
    params_s4[8] = 0.2  # f_ext_dy = 0.2 (some moderate side force)

    x0_scenario4 = x0_scenario1.copy()  # Start from the same initial conditions
    # Keep rotor speeds at hover to exactly balance the weight
    x0_scenario4[10] = hover_spin
    x0_scenario4[11] = hover_spin

    t_s4, x_s4 = simulate_system(
        x0_scenario4, (0, 5), params_s4, zero_control_law
    )
    plot_states(t_s4, x_s4, scenario_name="4) Side Wind/External Force on Drone")

    # -----------------------------------------------------------------
    # Scenario 5: Imbalanced rotor speeds => net torque -> rotation.
    # We still want total thrust = (M+m)*g, but let's set w1 != w2.
    # For instance:
    #    w1 + w2 = net_weight/c_
    #    w2 - w1 = some positive offset => net torque > 0
    #
    # This will cause the drone to rotate, while staying near same altitude.
    # We can watch the pendulum's behavior as the drone spins around.
    # -----------------------------------------------------------------
    total_w = net_weight / c_
    torque_offset = 0.3   # difference in rotor speed
    w1_imbal = 0.5*total_w - 0.5*torque_offset
    w2_imbal = 0.5*total_w + 0.5*torque_offset

    x0_scenario5 = x0_scenario1.copy()
    x0_scenario5[10] = w1_imbal  # w1
    x0_scenario5[11] = w2_imbal  # w2

    t_s5, x_s5 = simulate_system(
        x0_scenario5, (0, 5), params, zero_control_law
    )
    plot_states(t_s5, x_s5, scenario_name="5) Net Torque => Drone Rotation")
