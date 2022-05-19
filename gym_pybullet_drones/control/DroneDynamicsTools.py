import numpy as np
from scipy.linalg import block_diag
import control

def drone_dynamics(MASS, GRAV_ACCEL, IX, IY, IZ):
    """

    Parameters
    ----------
    MASS            - Drone mass
    GRAV_ACCEL      - g
    IX              - Moment of Inertia about X
    IY              - Moment of Inertia about Y
    IZ              - Moment of Inertia about Z

    Returns
    -------
    A               - Drone dynamics A matrix
    B               - Drone dynamics B matrix

    Both matrices follow the state definition:
        X = [x1, x1_dot, x2, x2_dot....]
    Thus, the state vector should be:
        X = [x, x_dot, y, y_dot, z, z_dot, r, r_dot, p, p_dot, y, y_dot]

    """
    # Form A matrix
    A_base = np.array([[0,1],[0,0]])
    A_cartesian = block_diag(A_base, A_base, A_base)
    A_angular = block_diag(A_base, A_base, A_base)

    # Form B matrices
    B_x = np.array([[0, GRAV_ACCEL]])
    B_y = np.array([[0, -GRAV_ACCEL]])
    B_z = np.array([[0, 1/MASS]])

    B_cartesian = block_diag(B_x.T, B_y.T, B_z.T)

    B_r = np.array([[0, 1/IX]])
    B_p = np.array([[0, 1/IY]])
    B_y = np.array([[0, 1/IZ]])

    B_angular = block_diag(B_r.T, B_p.T, B_y.T)

    return A_cartesian, B_cartesian, A_angular, B_angular

def euler_discretize(A,B,dt):
    return np.eye(np.shape(A))+dt*A, B


def solve_lqr_gains(A,B,Q,R):
    return control.lqr(A,B,Q,R)[0]

def shift_states(states, state_derivatives):
    # Shifts states to conform to the dynamics convention:
    # X = [x1, x1_dot, x2, x2_dot...]
    X = []
    for idx in range(3):
        X = np.append(X,[states[idx], state_derivatives[idx]])
    return X

def get_K_cart_hardcoded():
    """
    Hardcoded cartesian loop gains.
    """
    return np.array([[0.1205, 0.1598, 0, 0, 0, 0],
                     [0, 0, -.1205, -.1598, 0, 0],
                     [0, 0, 0, 0, 1.7502, -.5465]])/10

def get_K_ang_hardcoded():
    """
    Hardcoded attitude loop gains.
    """
    return np.array([[0.0040, 0.0009, 0, 0, 0, 0],
                     [0, 0, 0.0040, 0.0009, 0, 0],
                     [0, 0, 0, 0, 0.0034, 0.0013]])


def get_calculated_Ks(MASS, GRAV_ACCEL, IX, IY, IZ, Q_cartesian, R_cartesian, Q_angular, R_angular):
    """
    Computes continuous LQR gains within python given mass props and weighting matrices.
    """
    A_c, B_c, A_a, B_a = drone_dynamics(MASS, GRAV_ACCEL, IX, IY, IZ)
    K_cartesian = control.lqr(A_c,B_c, Q_cartesian, R_cartesian)[0]
    K_angular = control.lqr(A_a, B_a, Q_angular, R_angular)[0]
    return K_cartesian, K_angular

