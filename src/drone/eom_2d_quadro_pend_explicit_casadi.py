import casadi as ca

def eom_2d_quadro_pend_explicit(x, u, params):
    # x = [ y, z, phi, theta, y_dot, z_dot, phi_dot, theta_dot, w1, w2 ]
    # u = [u1, u2] = [dw1/dt, dw2/dt]
    # params = [M, m, Ixx, g, l0, c, L_rot, f_ext_dy, f_ext_dz, f_ext_my, f_ext_mz]

    params_list = ca.vertsplit(params)
    M = params_list[0]
    m = params_list[1]
    Ixx = params_list[2]
    g = params_list[3]
    l0 = params_list[4]
    c = params_list[5]
    L_rot = params_list[6]
    f_ext_dy = params_list[7]
    f_ext_dz = params_list[8]
    f_ext_my = params_list[9]
    f_ext_mz = params_list[10]

    x_dot = ca.MX.zeros(10)

    x_dot[0] = x[4]
    x_dot[1] = x[5]
    x_dot[2] = x[6]
    x_dot[3] = x[7]
    x_dot[4] = (-2*M*c*x[8]*ca.sin(x[2]) - 2*M*c*x[9]*ca.sin(x[2]) + 2*M*f_ext_dy - M*f_ext_my*ca.cos(2*x[3]) + M*f_ext_my - M*f_ext_mz*ca.sin(2*x[3]) + 2*M*l0*m*ca.sin(x[3])*x[7]**2 - c*m*x[8]*ca.sin(x[2] - 2*x[3]) - c*m*x[8]*ca.sin(x[2]) - c*m*x[9]*ca.sin(x[2] - 2*x[3]) - c*m*x[9]*ca.sin(x[2]) + f_ext_dy*m*ca.cos(2*x[3]) + f_ext_dy*m + f_ext_dz*m*ca.sin(2*x[3]))/(2*M*(M + m))
    x_dot[5] = (-2*M**2*g + 2*M*c*x[8]*ca.cos(x[2]) + 2*M*c*x[9]*ca.cos(x[2]) + 2*M*f_ext_dz - M*f_ext_my*ca.sin(2*x[3]) + M*f_ext_mz*ca.cos(2*x[3]) + M*f_ext_mz - 2*M*g*m - 2*M*l0*m*ca.cos(x[3])*x[7]**2 - c*m*x[8]*ca.cos(x[2] - 2*x[3]) + c*m*x[8]*ca.cos(x[2]) - c*m*x[9]*ca.cos(x[2] - 2*x[3]) + c*m*x[9]*ca.cos(x[2]) + f_ext_dy*m*ca.sin(2*x[3]) - f_ext_dz*m*ca.cos(2*x[3]) + f_ext_dz*m)/(2*M*(M + m))
    x_dot[6] = L_rot*c*(-x[8] + x[9])/Ixx
    x_dot[7] = (M*f_ext_my*ca.cos(x[3]) + M*f_ext_mz*ca.sin(x[3]) + c*m*x[8]*ca.sin(x[2] - x[3]) + c*m*x[9]*ca.sin(x[2] - x[3]) - f_ext_dy*m*ca.cos(x[3]) - f_ext_dz*m*ca.sin(x[3]))/(M*l0*m)
    x_dot[8] = u[0]
    x_dot[9] = u[1]
    return x_dot
