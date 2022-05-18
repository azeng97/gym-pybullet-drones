import math
import numpy as np
import pybullet as p

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.envs.BaseAviary import DroneModel, BaseAviary
from gym_pybullet_drones.utils.utils import nnlsRPM
from gym_pybullet_drones.control.DroneDynamicsTools import shift_states, get_K_cart, get_K_ang


class BasicLQRControl(BaseControl):

    def __init__(self,
                 drone_model: DroneModel,
                 g: float = 9.8
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
        self.MAX_ROLL_PITCH = np.pi / 6
        self.L = self._getURDFParameter('arm')
        self.THRUST2WEIGHT_RATIO = self._getURDFParameter('thrust2weight')
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO * self.GRAVITY) / (4 * self.KF))
        self.MAX_THRUST = (4 * self.KF * self.MAX_RPM ** 2)
        self.MAX_XY_TORQUE = (self.L * self.KF * self.MAX_RPM ** 2)
        self.MAX_Z_TORQUE = (2 * self.KM * self.MAX_RPM ** 2)
        self.A = np.array([[1, 1, 1, 1], [0, 1, 0, -1], [-1, 0, 1, 0], [-1, 1, -1, 1]])
        self.INV_A = np.linalg.inv(self.A)
        self.B_COEFF = np.array([1 / self.KF, 1 / (self.KF * self.L), 1 / (self.KF * self.L), 1 / self.KM])
        self.K_cartesian = get_K_cart()
        self.K_angular = get_K_ang()
        self.reset()

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()

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
        self.control_counter += 1
        cur_rpy = p.getEulerFromQuaternion(cur_quat)

        cartesian_states = shift_states(cur_pos, cur_vel)
        angular_states = shift_states(cur_rpy, cur_ang_vel)

        pos_err = cartesian_states - shift_states(target_pos, target_vel)

        # Use the externally computed gains to apply two loops of LQR.
        # Loop 1: Cartesian. Computes desired roll and pitch angles, as well as thrust.
        u_cartesian = self.K_cartesian @ pos_err

        # Unpack u_cartesian to provide:
        pitch_des = u_cartesian[0]
        roll_des = u_cartesian[1]
        thrust = u_cartesian[2] + self.GRAVITY

        computed_target_rpy = np.array([roll_des, pitch_des, 0, 0])

        # Loop 2: Angular. Computes the desired torques.
        rpy_err = angular_states - shift_states(computed_target_rpy, target_rpy_rates)
        target_torques = self.K_angular@rpy_err

        rpm = nnlsRPM(thrust=thrust,
                      x_torque=target_torques[0],
                      y_torque=target_torques[1],
                      z_torque=target_torques[2],
                      counter=self.control_counter,
                      max_thrust=self.MAX_THRUST,
                      max_xy_torque=self.MAX_XY_TORQUE,
                      max_z_torque=self.MAX_Z_TORQUE,
                      a=self.A,
                      inv_a=self.INV_A,
                      b_coeff=self.B_COEFF,
                      gui=True
                      )

        return rpm, computed_target_rpy
