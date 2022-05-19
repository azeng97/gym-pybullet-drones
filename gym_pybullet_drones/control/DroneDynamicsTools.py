import numpy as np
from scipy.linalg import block_diag
import control

def drone_dynamics(MASS, GRAV_ACCEL, IX, IY, IZ, USE_INTEGRAL_STATES=False, DISCRETIZE=False, dt = 1/48):
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
    if USE_INTEGRAL_STATES:
        A_base = np.array([[0, 1, 0], [0, 0, 0], [-1, 0, 0]])
    else:
        A_base = np.array([[0,1],[0,0]])

    A_cartesian = block_diag(A_base, A_base, A_base)
    A_angular = block_diag(A_base, A_base, A_base)

    # Form B matrices
    if USE_INTEGRAL_STATES:
        B_x = np.array([[0, GRAV_ACCEL, 0]])
        B_y = np.array([[0, -GRAV_ACCEL, 0]])
        B_z = np.array([[0, 1/MASS, 0]])
        B_r = np.array([[0, 1/IX, 0]])
        B_p = np.array([[0, 1/IY, 0]])
        B_y = np.array([[0, 1/IZ, 0]])
    else:
        B_x = np.array([[0, GRAV_ACCEL]])
        B_y = np.array([[0, -GRAV_ACCEL]])
        B_z = np.array([[0, 1/MASS]])
        B_r = np.array([[0, 1/IX]])
        B_p = np.array([[0, 1/IY]])
        B_y = np.array([[0, 1/IZ]])

    B_cartesian = block_diag(B_x.T, B_y.T, B_z.T)
    B_angular = block_diag(B_r.T, B_p.T, B_y.T)

    if DISCRETIZE:
        A_cartesian, B_cartesian = euler_discretize(A_cartesian, B_cartesian, dt)
        A_angular, B_angular = euler_discretize(A_angular, B_angular, dt)

    return A_cartesian, B_cartesian, A_angular, B_angular


def euler_discretize(A,B,dt):
    return np.eye(np.shape(A)[0])+dt*A, B


def solve_lqr_gains(A,B,Q,R):
    return control.lqr(A,B,Q,R)[0]


def shift_states(states, state_derivatives, state_integrals=None):
    # Shifts states to conform to the dynamics convention:
    # X = [x1, x1_dot, x2, x2_dot...]
    X = []
    for idx in range(3):
        if state_integrals is None:
            X = np.append(X, [states[idx], state_derivatives[idx]])
        else:
            X = np.append(X, [states[idx], state_derivatives[idx], state_integrals[idx]])
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


def get_calculated_Ks(MASS, GRAV_ACCEL, IX, IY, IZ, Q_cartesian, R_cartesian, Q_angular, R_angular, USE_INTEGRAL_STATES=False, DISCRETIZE=False, dt=10):
    """
    Computes continuous LQR gains within python given mass props and weighting matrices.
    """
    A_c, B_c, A_a, B_a = drone_dynamics(MASS, GRAV_ACCEL, IX, IY, IZ, USE_INTEGRAL_STATES, DISCRETIZE)

    K_cartesian = control.lqr(A_c,B_c, Q_cartesian, R_cartesian)[0]
    K_angular = control.lqr(A_a, B_a, Q_angular, R_angular)[0]
    return K_cartesian, K_angular
