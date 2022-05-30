import math
import numpy as np
import pybullet as pybul
import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.envs.BaseAviary import DroneModel, BaseAviary
from gym_pybullet_drones.utils.utils import nnlsRPM

from typing import Callable, NamedTuple
import matplotlib.pyplot as plt

class LinearDynamics(NamedTuple):
    f_x: jnp.array  # A
    f_u: jnp.array  # B

    def __call__(self, x, u, k=None):
        return self.f_x @ x + self.f_u @ u if k is None else self[k](x, u)

    def __getitem__(self, key):
        return jax.tree_map(lambda x: x[key], self)


class AffinePolicy(NamedTuple):
    l: jnp.array  # l
    l_x: jnp.array  # L

    def __call__(self, x, k=None):
        return self.l + self.l_x @ x if k is None else self[k](x)

    def __getitem__(self, key):
        return jax.tree_map(lambda x: x[key], self)


class QuadraticCost(NamedTuple):
    c: jnp.array  # c
    c_x: jnp.array  # q
    c_u: jnp.array  # r
    c_xx: jnp.array  # Q
    c_uu: jnp.array  # R
    c_ux: jnp.array  # H.T

    @classmethod
    def from_pure_quadratic(cls, c_xx, c_uu, c_ux):
        return cls(
            jnp.zeros((c_xx.shape[:-2])),
            jnp.zeros(c_xx.shape[:-1]),
            jnp.zeros(c_uu.shape[:-1]),
            c_xx,
            c_uu,
            c_ux,
        )

    def __call__(self, x, u, k=None):
        return (self.c + self.c_x @ x + self.c_u @ u + x @ self.c_xx @ x / 2 + u @ self.c_uu @ u / 2 +
                u @ self.c_ux @ x if k is None else self[k](x))

    def __getitem__(self, key):
        return jax.tree_map(lambda x: x[key], self)


class QuadraticStateCost(NamedTuple):
    v: jnp.array  # p (scalar)
    v_x: jnp.array  # p (vector)
    v_xx: jnp.array  # P

    @classmethod
    def from_pure_quadratic(cls, v_xx):
        return cls(
            jnp.zeros(v_xx.shape[:-2]),
            jnp.zeros(v_xx.shape[:-1]),
            v_xx,
        )

    def __call__(self, x, k=None):
        return self.v + self.v_x @ x + x @ self.v_xx @ x / 2 if k is None else self[k](x)

    def __getitem__(self, key):
        return jax.tree_map(lambda x: x[key], self)

def riccati_step(
    current_step_dynamics: LinearDynamics,
    current_step_cost: QuadraticCost,
    next_state_value: QuadraticStateCost,
):
    f_x, f_u = current_step_dynamics
    c, c_x, c_u, c_xx, c_uu, c_ux = current_step_cost
    v, v_x, v_xx = next_state_value

    q = c + v
    q_x = c_x + f_x.T @ v_x
    q_u = c_u + f_u.T @ v_x
    q_xx = c_xx + f_x.T @ v_xx @ f_x
    q_uu = c_uu + f_u.T @ v_xx @ f_u
    q_ux = c_ux + f_u.T @ v_xx @ f_x

    l = -jnp.linalg.solve(q_uu, q_u)
    l_x = -jnp.linalg.solve(q_uu, q_ux)

    current_state_value = QuadraticStateCost(
        q - l.T @ q_uu @ l / 2,
        q_x - l_x.T @ q_uu @ l,
        q_xx - l_x.T @ q_uu @ l_x,
    )
    current_step_optimal_policy = AffinePolicy(l, l_x)
    return current_state_value, current_step_optimal_policy


def rollout_state_feedback_policy(dynamics, policy, x0, step_range, x_nom=None, u_nom=None):

    def scan_fn(x, k):
        u = policy(x, k) if x_nom is None else u_nom[k] + policy(x - x_nom[k], k)
        x1 = dynamics(x, u, k)
        return (x1, (x1, u))

    xs, us = jax.lax.scan(scan_fn, x0, step_range)[1]
    return jnp.concatenate([x0[None], xs]), us

def ensure_positive_definite(a, eps=1e-3):
    return a + (eps - jnp.minimum(eps, jnp.linalg.eigvalsh(a)[0])) * jnp.eye(a.shape[-1])


class TotalCost(NamedTuple):
    running_cost: Callable
    terminal_cost: Callable

    def __call__(self, xs, us):
        step_range = jnp.arange(us.shape[0])
        return jnp.sum(jax.vmap(self.running_cost)(xs[:-1], us, step_range)) + self.terminal_cost(xs[-1])


@jax.jit
def iterative_linear_quadratic_regulator(dynamics, total_cost, x0, u_guess, maxiter=100, atol=1e-3):
    running_cost, terminal_cost = total_cost
    n, (N, m) = x0.shape[-1], u_guess.shape
    step_range = jnp.arange(N)

    xs_iterates, us_iterates = jnp.zeros((maxiter, N + 1, n)), jnp.zeros((maxiter, N, m))
    xs, us = rollout_state_feedback_policy(dynamics, lambda x, k: u_guess[k], x0, step_range)
    xs_iterates, us_iterates = xs_iterates.at[0].set(xs), us_iterates.at[0].set(us)
    j_curr = total_cost(xs, us)
    value_functions_iterates = QuadraticStateCost.from_pure_quadratic(jnp.zeros((maxiter, N + 1, n, n)))

    def continuation_criterion(loop_vars):
        i, _, _, j_curr, j_prev, _ = loop_vars
        return (j_curr < j_prev - atol) & (i < maxiter)

    def ilqr_iteration(loop_vars):
        i, xs_iterates, us_iterates, j_curr, j_prev, value_functions_iterates = loop_vars
        xs, us = xs_iterates[i], us_iterates[i]

        f_x, f_u = jax.vmap(jax.jacobian(dynamics, (0, 1)))(xs[:-1], us, step_range)
        c = jax.vmap(running_cost)(xs[:-1], us, step_range)
        c_x, c_u = jax.vmap(jax.grad(running_cost, (0, 1)))(xs[:-1], us, step_range)
        (c_xx, c_xu), (c_ux, c_uu) = jax.vmap(jax.hessian(running_cost, (0, 1)))(xs[:-1], us, step_range)
        v, v_x, v_xx = terminal_cost(xs[-1]), jax.grad(terminal_cost)(xs[-1]), jax.hessian(terminal_cost)(xs[-1])

        # Ensure quadratic cost terms are positive definite.
        c_zz = jnp.block([[c_xx, c_xu], [c_ux, c_uu]])
        c_zz = jax.vmap(ensure_positive_definite)(c_zz)
        c_xx, c_uu, c_ux = c_zz[:, :n, :n], c_zz[:, -m:, -m:], c_zz[:, -m:, :n]
        v_xx = ensure_positive_definite(v_xx)

        linearized_dynamics = LinearDynamics(f_x, f_u)
        quadratized_running_cost = QuadraticCost(c, c_x, c_u, c_xx, c_uu, c_ux)
        quadratized_terminal_cost = QuadraticStateCost(v, v_x, v_xx)

        def scan_fn(next_state_value, current_step_dynamics_cost):
            current_step_dynamics, current_step_cost = current_step_dynamics_cost
            current_state_value, current_step_policy = riccati_step(
                current_step_dynamics,
                current_step_cost,
                next_state_value,
            )
            return current_state_value, (current_state_value, current_step_policy)

        value_functions, policy = jax.lax.scan(scan_fn,
                                               quadratized_terminal_cost,
                                               (linearized_dynamics, quadratized_running_cost),
                                               reverse=True)[1]
        value_functions_iterates = jax.tree_map(lambda x, xi, xiN: x.at[i].set(jnp.concatenate([xi, xiN[None]])),
                                                value_functions_iterates, value_functions, quadratized_terminal_cost)

        def rollout_linesearch_policy(alpha):
            # Note that we roll out the true `dynamics`, not the `linearized_dynamics`!
            return rollout_state_feedback_policy(dynamics, AffinePolicy(alpha * policy.l, policy.l_x), x0, step_range,
                                                 xs, us)

        # Backtracking line search (step sizes evaluated in parallel).
        all_xs, all_us = jax.vmap(rollout_linesearch_policy)(0.5**jnp.arange(8))
        js = jax.vmap(total_cost)(all_xs, all_us)
        a = jnp.argmin(js)
        j = js[a]
        xs_iterates = xs_iterates.at[i + 1].set(jnp.where(j < j_curr, all_xs[a], xs))
        us_iterates = us_iterates.at[i + 1].set(jnp.where(j < j_curr, all_us[a], us))
        return i + 1, xs_iterates, us_iterates, jnp.minimum(j, j_curr), j_curr, value_functions_iterates

    i, xs_iterates, us_iterates, j_curr, j_prev, value_functions_iterates = jax.lax.while_loop(
        continuation_criterion, ilqr_iteration,
        (0, xs_iterates, us_iterates, j_curr, jnp.inf, value_functions_iterates))

    return {
        "optimal_trajectory": (xs_iterates[i], us_iterates[i]),
        "optimal_cost": j_curr,
        "num_iterations": i,
        "trajectory_iterates": (xs_iterates, us_iterates),
        "value_functions_iterates": value_functions_iterates
    }

class RK4Integrator(NamedTuple):
    """Discrete time dynamics from time-invariant continuous time dynamics using a 4th order Runge-Kutta method."""
    ode: Callable
    dt: float

    def __call__(self, x, u, k):
        k1 = self.dt * self.ode(x, u)
        #print(self.ode(x,u))
        k2 = self.dt * self.ode(x + k1 / 2, u)
        k3 = self.dt * self.ode(x + k2 / 2, u)
        k4 = self.dt * self.ode(x + k3, u)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6


class DroneDynamics(NamedTuple):
    g: float
    m: float 
    Ix: float
    Iy: float
    Iz: float

    def __call__(self, state, control):
        g = self.g
        m = self.m
        Ix = self.Ix
        Iy = self.Iy
        Iz = self.Iz

        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state
        u1, u2, u3, u4 = control

        # R_BI = jnp.zeros((3, 3))
        # R_BI = R_BI.at[0,0].set(jnp.cos(theta)*jnp.cos(psi))
        # R_BI[0, 1] = jnp.cos(theta)*jnp.sin(psi)
        # R_BI[0, 2] = -jnp.sin(theta)

        # R_BI[1, 0] = jnp.sin(theta)*jnp.cos(psi)*jnp.sin(phi) - jnp.sin(psi)*jnp.cos(phi)
        # R_BI[1, 1] = jnp.sin(theta)*jnp.sin(psi)*jnp.sin(phi) + jnp.cos(psi)*jnp.cos(phi)
        # R_BI[1, 2] = jnp.cos(theta)*jnp.sin(phi)

        # R_BI[2, 0] = jnp.sin(theta)*jnp.cos(psi)*jnp.cos(phi) + jnp.sin(psi)*jnp.sin(phi)
        # R_BI[2, 1] = jnp.sin(theta)*jnp.sin(psi)*jnp.cos(phi) - jnp.cos(psi)*jnp.sin(phi)
        # R_BI[2, 2] = jnp.cos(theta)*jnp.cos(phi)

        R_BI = jnp.array([ [jnp.cos(theta)*jnp.cos(psi),jnp.cos(theta)*jnp.sin(psi), -jnp.sin(theta)],
                            [jnp.sin(theta)*jnp.cos(psi)*jnp.sin(phi) - jnp.sin(psi)*jnp.cos(phi),jnp.sin(theta)*jnp.sin(psi)*jnp.sin(phi) + jnp.cos(psi)*jnp.cos(phi), jnp.cos(theta)*jnp.sin(phi)],
                            [jnp.sin(theta)*jnp.cos(psi)*jnp.cos(phi) + jnp.sin(psi)*jnp.sin(phi),jnp.sin(theta)*jnp.sin(psi)*jnp.cos(phi) - jnp.cos(psi)*jnp.sin(phi),jnp.cos(theta)*jnp.cos(phi)]])

        # W = jnp.zeros((3,3))
        # W[0,0] = 1
        # W[0,1] = jnp.sin(phi)*jnp.tan(theta)
        # W[0,2] = jnp.cos(phi)*jnp.tan(theta)

        # W[1,0] = 0
        # W[1,1] = jnp.cos(phi)
        # W[1,2] = -jnp.sin(phi)

        # W[2,0] = 0
        # W[2,1] = jnp.sin(phi)*jnp.arccos(theta)
        # W[2,2] = jnp.cos(theta)*jnp.arccos(theta)

        W = jnp.array([[1, jnp.sin(phi)*jnp.tan(theta), jnp.cos(phi)*jnp.tan(theta)],
                        [0, jnp.cos(phi), -jnp.sin(phi)],
                        [0, jnp.sin(phi)*jnp.arccos(theta), jnp.cos(theta)*jnp.arccos(theta)]])

        # J = jnp.zeros((3,3))
        # J[0,0] = Ix 
        # J[1,1] = Iy 
        # J[2,2] = Iz

        J = jnp.array([[Ix, 0, 0],
                      [0, Iy, 0],
                      [0, 0, Iz]])

        #e3 = jnp.zeros((3,1))
        e3 = jnp.array([0, 0, 1]).T

        Vb = jnp.array([vx, vy, vz]).T
        OMEGA = jnp.array([p, q, r]).T
        TAU = jnp.array([u2, u3, u4]).T

        return jnp.hstack((jnp.dot(R_BI.T, Vb),
                        -jnp.cross(OMEGA, Vb) + jnp.dot(R_BI, -g*e3) + u1 / m * e3,
                        jnp.dot(W,OMEGA),
                        jnp.dot(jnp.linalg.inv(J), -jnp.cross(OMEGA, jnp.dot(J, OMEGA)) + TAU))) 

class VehicleExampleRunningCost(NamedTuple):
    soft_u1_limit: float
    target_pos: np.ndarray
    target_rpy: np.ndarray
    target_vel: np.ndarray
    target_rpy_rates: np.ndarray

    def __call__(self, state, control, step):
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state
        u1, u2, u3, u4 = control
        target_x, target_y, target_z = self.target_pos
        target_phi, target_th, target_psi = self.target_rpy
        target_p, target_q, target_r = self.target_rpy_rates


        if True:
            target_x, target_y, target_z = self.target_pos
            target_phi, target_th, target_psi = self.target_rpy
            # target_psi_rate = self.target_rpy_rates[2]
            target_psi_rate = 0

            lanekeeping = 1*(phi**2 + theta**2 + p**2 + q**2 + (r - target_psi_rate)**2)
            #avoid_origin = 10000 * jnp.exp((-x**2 - y**2) / 10)
            #acceleration_cost = a**2 + a_lat**2
            u1_limit = 100 * jnp.maximum(u1**2 - self.soft_u1_limit**2, 0)
            u1_limit_neg = 10 * jnp.maximum(jnp.sign(u1)*u1**2, 0)
            theta_limit = 0#1000 * jnp.maximum(theta**2 - (jnp.pi/2)**2, 0)
            # target_pose = 1*((z - 0.75)**2) + 0.001*((x+0.5)**2 + (y+0.5)**2 + (psi - jnp.pi/4)**2) #+ 100*(theta - jnp.pi/8)**2
            target_pose = 1*((z - target_z)**2) + 0.001*(10*(x - target_x)**2 + 10*(y - target_y)**2 + (target_psi - jnp.pi/4)**2) #+ 100*(theta - jnp.pi/8)**2
            #print(theta)

            return lanekeeping + u1_limit + u1_limit_neg + theta_limit + 5*(u1**2) + 50000*(u2**2 + u3**2 + u4**2) + target_pose
            #return 0.0
        else:
            # Krt tests
            psi_e = jnp.minimum(psi-target_psi-2*jnp.pi, psi-target_psi+2*jnp.pi)

            pos_state_cost = 1*((x-target_x)**2 + (y-target_y)**2 + 5*(z-target_z)**2)                                   # Penalize distance from target pos
            vel_state_cost = 0
            ang_state_cost = (phi-target_phi)**2 + 10*(theta-target_th)**2 + 10*(psi_e)**2

            # Penalize any angular rates to ensure smooth trajectory
            ang_rate_state_cost = 10*(p-target_p)**2 + 10*(q-target_q)**2 + (r-target_r)**2

            # Penalize torques more than thrust
            control_cost= 5*u1**2 + 5000*(u2**2 + u3**2 + u4**2)

            return control_cost + pos_state_cost + ang_state_cost + p**2 + r**2 + q**2


class VehicleExampleTerminalCost(NamedTuple):
    target_pos: np.ndarray
    target_rpy: np.ndarray
    target_vel: np.ndarray
    target_rpy_rates: np.ndarray
    gain: float = 1.0

    def __call__(self, state):
        target_x, target_y, target_z = self.target_pos
        target_phi, target_th, target_psi = self.target_rpy

        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state

        if True:
            # Original (100x gain)
            return 100*((z - target_z)**2 + (x - target_x)**2 +(y - target_y)**2 + vz**2 + vx**2 + vy**2 + phi**2 + theta**2 + p**2 + q**2 + r**2 + (target_psi - jnp.pi/4)**2)
        else:
            # krt
            return 1e4*((x-target_x)**2 + (y-target_y)**2 + (z-target_z)**2)

class ILQRControl(BaseControl):
    """Generic PID control class without yaw control.

    Based on https://github.com/prfraanje/quadcopter_sim.

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 control_freq: int,
                 g: float=9.8,
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        # if self.DRONE_MODEL != DroneModel.HB:
        #     print("[ERROR] in MPCControl.__init__(), MPCControl requires DroneModel.HB")
        #     exit()
        self.DRONE_MODEL = drone_model
        self.I_X = self._getURDFParameter('ixx')    # Moments of inertia
        self.I_Y = self._getURDFParameter('iyy')
        self.I_Z = self._getURDFParameter('izz')
        self.M = self._getURDFParameter('m')        # Mass
        self.G = g

        self.dynamics = DroneDynamics(g, self.M, self.I_X, self.I_Y, self.I_Z)

                                      # Gravitational constant

        # self.P_COEFF_FOR = np.array([.1, .1, .2])
        # self.I_COEFF_FOR = np.array([.0001, .0001, .0001])
        # self.D_COEFF_FOR = np.array([.3, .3, .4])
        # self.P_COEFF_TOR = np.array([.3, .3, .05])
        # self.I_COEFF_TOR = np.array([.0001, .0001, .0001])
        # self.D_COEFF_TOR = np.array([.3, .3, .5])
        self.MAX_ROLL_PITCH = np.pi/6
        self.L = self._getURDFParameter('arm')
        self.THRUST2WEIGHT_RATIO = self._getURDFParameter('thrust2weight')
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (4*self.KF))
        self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
        #print(self.MAX_RPM)
        #print(self.KF)
        self.MAX_XY_TORQUE = (self.L*self.KF*self.MAX_RPM**2)
        self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)
        self.A = np.array([ [1, 1, 1, 1], [0, 1, 0, -1], [-1, 0, 1, 0], [-1, 1, -1, 1] ])
        self.INV_A = np.linalg.inv(self.A)
        self.B_COEFF = np.array([1/self.KF, 1/(self.KF*self.L), 1/(self.KF*self.L), 1/self.KM])
        self.control_freq = control_freq  # hz
        self.horizon = 1  # seconds to plan (orig = 1)
        self.reset()


        self.u_init = np.zeros((int(self.horizon*self.control_freq), 4))
        self.u_init[:,0] = self.M * self.G * self.u_init[:,0]

    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)
    
      
    ################################################################################


    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """Computes the PID control action (as RPMs) for a single drone.

        This methods sequentially calls `_simplePIDPositionControl()` and `_simplePIDAttitudeControl()`.
        Parameters `cur_ang_vel`, `target_rpy`, `target_vel`, and `target_rpy_rates` are unused.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        self.control_counter += 1
        if target_rpy[2]!=0:
            print("\n[WARNING] ctrl it", self.control_counter, "in SimplePIDControl.computeControl(), desired yaw={:.0f}deg but locked to 0. for DroneModel.HB".format(target_rpy[2]*(180/np.pi)))

        cur_rpy = pybul.getEulerFromQuaternion(cur_quat)
        x0 = np.hstack((cur_pos, cur_vel, cur_rpy, cur_ang_vel))
        
        #x0 = np.hstack((cur_pos[2], cur_vel[2]))
        #print(x0)
        solution = iterative_linear_quadratic_regulator(
            RK4Integrator(self.dynamics, 1/self.control_freq),
            TotalCost(
                VehicleExampleRunningCost(soft_u1_limit=self.MAX_THRUST,
                                          target_pos=target_pos,
                                          target_rpy=target_rpy,
                                          target_vel=target_vel,
                                          target_rpy_rates=target_rpy_rates),
                VehicleExampleTerminalCost(gain=1.0,
                                           target_pos=target_pos,
                                           target_rpy=target_rpy,
                                           target_vel=target_vel,
                                           target_rpy_rates=target_rpy_rates),
            ),
            x0,
            self.u_init,
        )
        states, controls = solution["optimal_trajectory"]
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = states.T
        u1, u2, u3, u4 = controls.T

        self.u_init = controls

        print(RK4Integrator(self.dynamics, 1/self.control_freq)(x0, np.array([u1[0], u2[0], u3[0], u4[0]]), 0))

        u_final = nnlsRPM(thrust=u1[0],
                          x_torque=u2[0],
                          y_torque=u3[0],
                          z_torque=u4[0],
                          counter=self.control_counter,
                          max_thrust=self.MAX_THRUST,
                          max_xy_torque=self.MAX_XY_TORQUE,
                          max_z_torque=self.MAX_Z_TORQUE,
                          a=self.A,
                          inv_a=self.INV_A,
                          b_coeff=self.B_COEFF,
                          gui=True
                          )

        print("thrust and torques:")
        print(np.array([u1[0], u2[0], u3[0], u4[0]]))
        print("rpms:")
        print(u_final)

        debug = False
        if debug:# or np.isnan(solution['optimal_cost']):
            plt.figure()
            plt.subplot(3,2,1)
            plt.plot(x, label="x")
            plt.plot(y, label='y')
            plt.plot(z,  label='z')
            plt.legend()
            plt.subplot(3,2,2)
            plt.plot(vx,  label='vx')
            plt.plot(vy,  label='vy')
            plt.plot(vz,  label='vz')
            plt.legend()
            plt.subplot(3,2,3)
            plt.plot(phi,  label='phi')
            plt.plot(theta,  label='theta')
            plt.plot(psi,  label='psi')
            plt.legend()
            plt.subplot(3,2,4)
            plt.plot(p,  label='p')
            plt.plot(q,  label='q')
            plt.plot(r,  label='r')
            plt.legend()
            plt.subplot(3,2,5)
            plt.plot(u1, label="u1")
            plt.legend()
            plt.subplot(3,2,6)
            plt.plot(u2, label="u2")
            plt.plot(u3, label="u3")
            plt.plot(u4, label="u4")
            plt.legend()

        print(solution['optimal_cost'])
        plt.show()

        # if self.control_counter > 2:
        #   sys.exit()
        return u_final, None, None    # dummy vars to match outputs of PID

    ################################################################################
