"""
Contains the dynamic model and transition functionality for the AA203 Final Project.
Dynamic model from [1] - Okyere et. al. "LQR Controller design for quad-rotor helicopters."

Assumes zero motor inertia for all motors. (J=\vec{0})
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
import xml.etree.ElementTree as etxml
from gym_pybullet_drones.envs.BaseAviary import DroneModel
from typing import Callable, NamedTuple



class Dynamics(object):

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        self.DRONE_MODEL = drone_model
        self.I_X = self._getURDFParameter('ixx')    # Moments of inertia
        self.I_Y = self._getURDFParameter('iyy')
        self.I_Z = self._getURDFParameter('izz')
        self.M = self._getURDFParameter('m')        # Mass
        self.G = g                                  # Gravitational constant

    def drone_dynamics(self, state, control):
        """
        Input: state - State vector. Defined as follows:
                       X = [x,y,z,r,p,y,x_d,y_d,z_d,r_d,p_d,y_d]
                       ... where r = roll, p = pitch, y = yaw,
                       and the _d suffix denotes a time derivative.

               control - Control inputs for the system as defined in [1].
                       U = [thrust, torque 1, torque 2, torque 3] = [U1, U2, U3, U4]

        Outputs the time derivative of the state vector.

        """
        g = self.G
        m = self.M
        Ix = self.I_X
        Iy = self.I_Y
        Iz = self.I_Z

        x, y, z, r, p, y, xd, yd, zd, rd, pd, yd = state
        u1, u2, u3, u4 = control
        state_dot = np.array([xd,
                            yd,
                            zd,
                            rd,
                            pd,
                            yd,
                            (np.cos(r) * np.sin(p) * np.cos(y) + np.sin(r) * np.sin(y)) * u1 / m,
                            (np.cos(r) * np.sin(p) * np.sin(y) - np.sin(r) * np.sin(y)) * u1 / m,
                            g - np.cos(r) * np.cos(p) * u1 / m,
                            pd * yd * (Ix - Iz) / Ix + u2 / Ix,
                            rd * yd * (Iz - Ix) / Iy + u3 / Iy,
                            rd * pd * (Ix - Iy) / Iz + u4 / Iz])
        return state_dot

    def step(self, state, control, dt):
        """
        A simple Euler discretization of the nonlinear drone dynamics.

        Inputs: state, control - See above fcn "drone_dynamics"
                dt             - timestep.
        Outputs: State at next timestep.
        """

        return state+dt*self.drone_dynamics(state, control)

    # Shamelessly taken from "BaseControl.py" to get drone data.
    def _getURDFParameter(self,
                          parameter_name: str
                          ):
        """Reads a parameter from a drone's URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        Parameters
        ----------
        parameter_name : str
            The name of the parameter to read.

        Returns
        -------
        float
            The value of the parameter.

        """
        #### Get the XML tree of the drone model to control ########
        URDF = self.DRONE_MODEL.value + ".urdf"
        URDF_TREE = etxml.parse(os.path.dirname(os.path.abspath(__file__))+"/../assets/"+URDF).getroot()
        #### Find and return the desired parameter #################
        if parameter_name == 'm':
            return float(URDF_TREE[1][0][1].attrib['value'])
        elif parameter_name in ['ixx', 'iyy', 'izz']:
            return float(URDF_TREE[1][0][2].attrib[parameter_name])
        elif parameter_name in ['arm', 'thrust2weight', 'kf', 'km', 'max_speed_kmh', 'gnd_eff_coeff' 'prop_radius', \
                                'drag_coeff_xy', 'drag_coeff_z', 'dw_coeff_1', 'dw_coeff_2', 'dw_coeff_3']:
            return float(URDF_TREE[0].attrib[parameter_name])
        elif parameter_name in ['length', 'radius']:
            return float(URDF_TREE[1][2][1][0].attrib[parameter_name])
        elif parameter_name == 'collision_z_offset':
            COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
            return COLLISION_SHAPE_OFFSETS[2]
