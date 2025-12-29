import casadi as ca

def eom_2d_quadro_springpend_explicit(x, u, params):
    # x = [ y, z, phi, r, theta, y_dot, z_dot, phi_dot, r_dot, theta_dot, w1, w2 ]
    # u = [u1, u2] = [dw1/dt, dw2/dt]
    # params = [M, m, Ixx, g, k, l0, c, L_rot, f_ext_dy, f_ext_dz, f_ext_my, f_ext_mz]

    params_list = ca.vertsplit(params)
    M = params_list[0]
    m = params_list[1]
    Ixx = params_list[2]
    g = params_list[3]
    k = params_list[4]
    l0 = params_list[5]
    c = params_list[6]
    L_rot = params_list[7]
    f_ext_dy = params_list[8]
    f_ext_dz = params_list[9]
    f_ext_my = params_list[10]
    f_ext_mz = params_list[11]

    x_dot = ca.MX.zeros(12)

    x_dot[0] = x[5]
    x_dot[1] = x[6]
    x_dot[2] = x[7]
    x_dot[3] = x[8]
    x_dot[4] = x[9]
    x_dot[5] = (-c*x[10]*ca.sin(x[2]) - c*x[11]*ca.sin(x[2]) + f_ext_dy - k*l0*ca.sin(x[4]) + k*x[3]*ca.sin(x[4]))/M
    x_dot[6] = (-M*g + c*x[10]*ca.cos(x[2]) + c*x[11]*ca.cos(x[2]) + f_ext_dz + k*l0*ca.cos(x[4]) - k*x[3]*ca.cos(x[4]))/M
    x_dot[7] = L_rot*c*(-x[10] + x[11])/Ixx
    x_dot[8] = f_ext_my*ca.sin(x[4])/m - f_ext_mz*ca.cos(x[4])/m + k*l0/m - k*x[3]/m + x[3]*x[9]**2 + c*x[10]*ca.cos(x[2] - x[4])/M + c*x[11]*ca.cos(x[2] - x[4])/M - f_ext_dy*ca.sin(x[4])/M + f_ext_dz*ca.cos(x[4])/M + k*l0/M - k*x[3]/M
    x_dot[9] = (1.0*M*f_ext_my*ca.cos(x[4]) + 1.0*M*f_ext_mz*ca.sin(x[4]) - 2.0*M*m*x[8]*x[9] + 1.0*c*m*x[10]*ca.sin(x[2] - x[4]) + 1.0*c*m*x[11]*ca.sin(x[2] - x[4]) - 1.0*f_ext_dy*m*ca.cos(x[4]) - 1.0*f_ext_dz*m*ca.sin(x[4]))/(M*m*(ca.sign(x[3]) * ca.fabs(x[3]) + 1e-6)) # added the devision by zero prevention manually
    x_dot[10] = u[0]
    x_dot[11] = u[1]
    return x_dot
