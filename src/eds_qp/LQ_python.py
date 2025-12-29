"""
Functionality to computes the sensitivity decay for an unconstrained QP (Figure 5 in the paper appendix)
"""

import numpy as np
from scipy.linalg import block_diag

def generate_random_system_matrices(n, m):
    A = np.random.randn(n, n)
    B = np.random.randn(n, m)
    return A, B

def sensitivities_closed_form(A, B, Q, R, P, N, n, m):
    Q_bar = block_diag(*([Q] * (N - 1) + [P]))
    R_bar = block_diag(*([R] * N))
    
    # Build A_bar and B_bar
    A_bar = []
    B_bar = []

    for i in range(1, N + 1):
        A_power = np.linalg.matrix_power(A, i) 
        A_bar.append(A_power)

        B_row = []
        for j in range(1, N + 1):
            if j <= i:
                B_row.append(np.linalg.matrix_power(A, i - j) @ B)
            else:
                B_row.append(np.zeros((n, m)))
        B_bar.append(np.hstack(B_row))
    
    A_bar = np.vstack(A_bar)
    B_bar = np.vstack(B_bar)

    # Build E matrix
    E = np.zeros((N * n, N * n))
    for i in range(N):
        for j in range(i + 1):
            E[i*n:(i+1)*n, j*n:(j+1)*n] = np.linalg.matrix_power(A, i - j)

    # Sensitivity formula
    H = B_bar.T @ Q_bar @ B_bar + R_bar
    G = B_bar.T @ Q_bar @ E
    sensitivities = -np.linalg.solve(H, G)
    return sensitivities
