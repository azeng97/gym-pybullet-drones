import jax.numpy as jnp
from typing import Callable, NamedTuple


# From Stanford ASL's Github
class RK4Integrator(NamedTuple):
    """Discrete time dynamics from time-invariant continuous time dynamics using a 4th order Runge-Kutta method."""
    ode: Callable
    dt: float

    def __call__(self, x, u, k):
        k1 = self.dt * self.ode(x, u)
        k2 = self.dt * self.ode(x + k1 / 2, u)
        k3 = self.dt * self.ode(x + k2 / 2, u)
        k4 = self.dt * self.ode(x + k3, u)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def drone_dynamics(state, control, params):
    """
    Returns the continuous time state derivative of a 6DOF quadrotor as a function of state and control.
    Model from: "LQR Controller Design for Quad-Rotor Helicopters" Assumes zero rotor inertia (J=0)

    Notation:
        r,p,y = roll, pitch, yaw.
        _d denotes a derivative state.
    """

    x, y, z, x_d, y_d, z_d, r, p, y, r_d, p_d, y_d = state
    u1, u2, u3, u4 = control
    m, ixx, iyy, izz, g = params

    state_derivative = jnp.array([ x_d,
                                   y_d,
                                   z_d,
                                   (jnp.cos(r)*jnp.sin(p)*jnp.cos(y)+jnp.sin(r)*jnp.sin(y))/m*u1,
                                   (jnp.cos(r)*jnp.sin(p)*jnp.sin(y)-jnp.sin(r)*jnp.sin(y))/m*u1,
                                   g-(jnp.cos(r)*jnp.cos(p))/m*u1,
                                   r_d,
                                   p_d,
                                   y_d,
                                   p_d*y_d*((ixx-izz)/ixx)+u2/ixx,
                                   r_d*y_d*((izz-ixx)/iyy)+u3/iyy,
                                   r_d*p_d*((ixx-iyy)/izz)+u4/izz
                                   ])

