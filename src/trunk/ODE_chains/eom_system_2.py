import numpy as np

def eom_2(q, q_dot, q_ddot, u, params):
    g, m, l, Iz, c, d, gear = params
    eqs = np.array([
        Iz*(2*q_ddot[0] + q_ddot[1]) + c*q[0] + d*q_dot[0] + g*l*m*(3*np.sin(q[0]) + np.sin(q[0] + q[1])) - gear*u[0] + l**2*m*(4*q_ddot[0]*np.cos(q[1]) + 6*q_ddot[0] + 2*q_ddot[1]*np.cos(q[1]) + q_ddot[1] - 4*q_dot[0]*q_dot[1]*np.sin(q[1]) - 2*q_dot[1]**2*np.sin(q[1])),
        Iz*q_ddot[0] + Iz*q_ddot[1] + c*q[1] + d*q_dot[1] + g*l*m*np.sin(q[0] + q[1]) - gear*u[1] + 2*l**2*m*q_ddot[0]*np.cos(q[1]) + l**2*m*q_ddot[0] + l**2*m*q_ddot[1] + 2*l**2*m*q_dot[0]**2*np.sin(q[1]),
    ])
    return eqs
