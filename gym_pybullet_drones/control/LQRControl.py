import numpy as np
import pybullet as p
import control
from scipy.linalg import block_diag

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.envs.BaseAviary import DroneModel, BaseAviary
from gym_pybullet_drones.utils.utils import nnlsRPM

"""
Implements the dynamic and LQR controller design methodology presented in:

[1] Martins et. al. - "Linear Quadratic Regulator for Trajectory Tracking of a Quadrotor"
    https://www.sciencedirect.com/science/article/pii/S2405896319311450

"""

class LQRControl(BaseControl):
    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8):
        super().__init__(drone_model=drone_model,g=g)

        self.MAX_ROLL_PITCH = np.pi/6
        self.L = self._getURDFParameter('arm')
        self.THRUST2WEIGHT_RATIO = self._getURDFParameter('thrust2weight')
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (4*self.KF))
        self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
        self.MAX_XY_TORQUE = (self.L*self.KF*self.MAX_RPM**2)
        self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)
        self.A = np.array([ [1, 1, 1, 1], [0, 1, 0, -1], [-1, 0, 1, 0], [-1, 1, -1, 1] ])
        self.INV_A = np.linalg.inv(self.A)
        self.B_COEFF = np.array([1/self.KF, 1/(self.KF*self.L), 1/(self.KF*self.L), 1/self.KM])

        self.MASS = self._getURDFParameter('m')
        self.I_x = self._getURDFParameter('ixx')
        self.I_y = self._getURDFParameter('iyy')
        self.I_z = self._getURDFParameter('izz')
        self.GRAV_ACCEL = g

        # Set Q, R, matrices for each subsystem.
        self.Q_cartesian = block_diag(25,1,25,1,25,1)
        self.R_cartesian = block_diag(1000,1000,3)
        self.Q_angular = block_diag(250,1,250,1,10,1)
        self.R_angular = block_diag(200,200,1)

        self._computeLQRGains()
        self.reset()

    def reset(self):
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
        r_des, p_des, thrust = self._positionControl(cur_pos, cur_vel, target_pos, target_vel)
        target_rpy = np.array([r_des, p_des, 0.0])
        target_torques = self._attitudeControl(cur_quat, cur_ang_vel, target_rpy, target_rpy_rates)
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

        # Set output vars
        pos_e = target_pos - cur_pos
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        yaw_e = target_rpy[2] - cur_rpy[2]
        return rpm, pos_e, yaw_e

    def _positionControl(self,
                         cur_pos,
                         cur_vel,
                         target_pos,
                         target_vel):
        x = self._shiftState(cur_pos, cur_vel)
        delta_x = self._shiftState(target_pos, target_vel)-x
        u = self.K_cartesian@delta_x

        # Unpack for clarity
        r_des = np.clip(u[0], -self.MAX_ROLL_PITCH, self.MAX_ROLL_PITCH)
        p_des = np.clip(u[1], -self.MAX_ROLL_PITCH, self.MAX_ROLL_PITCH)
        T = u[2] + self.GRAVITY
        return r_des, p_des, T


    def _attitudeControl(self,
                         cur_quat,
                         cur_ang_vel,
                         target_rpy,
                         target_rpy_rates):
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        x = self._shiftState(cur_rpy, cur_ang_vel)
        delta_x = self._shiftState(target_rpy, target_rpy_rates) - x
        u = self.K_angular@delta_x
        return u

    def _shiftState(self, x, x_dot):
        """
        Input: State, State Derivative (3,) and (3,)
        Output: Reshaped state vector according to [x1, x1_dot, x2, x2_dot...] convention. (6,)
        """
        x_new = []
        for i in range(3):
            x_new = np.append(x_new, [x[i], x_dot[i]])
        return x_new


    def _computeLQRGains(self):
        """
        Cartesian dynamics as specified in [1]

        A_cartesian - the A matrix for the planar system dynamics.
                State vector for the cartesian dynamics is defined as:
                X = [x, x_dot, y, y_dot, z, z_dot]
        B_cartesian - the B matrix for the planar system dynamics.
                Control inputs are modified such that:
                U = [u_x, u_y, u_z]
                where,
                    u_x = pitch_desired (theta)
                    u_y = roll_desired (phi)
                    u_z = T-mg (thrust minus weight)
        """
        A_base = np.array([[0, 1], [0, 0]])
        self.A_cartesian = block_diag(A_base, A_base, A_base)

        B_x = np.array([[0, self.GRAV_ACCEL]])
        B_y = np.array([[0, -self.GRAV_ACCEL]])
        B_z = np.array([[0, 1/self.MASS]])
        self.B_cartesian = block_diag(B_x.T, B_y.T, B_z.T)

        """
        Angular dynamics as defined in [1]
        A_angular - The A matrix for the attitude dynamics
            State vector for the angular dynamics is defined as:
                X = [phi, phi_dot, theta, theta_dot, psi, psi_dot]
                (phi = roll, theta = pitch, psi = yaw)
        B_angular - The B matrix for the attitude dynamics
            Control inputs are modified such that they correspond to torques as follows:
                U = [u_phi, u_theta, u_psi] = [Tau_phi, Tau_theta, Tau_psi]
        """
        self.A_angular = block_diag(A_base, A_base, A_base)

        B_roll = np.array([[0, 1/self.I_x]])
        B_pitch = np.array([[0, 1/self.I_y]])
        B_yaw = np.array([[0, 1/self.I_z]])

        self.B_angular = block_diag(B_roll.T, B_pitch.T, B_yaw.T)

        self.K_angular = control.lqr(self.A_angular, self.B_angular, self.Q_angular, self.R_angular)[0]
        self.K_cartesian = control.lqr(self.A_cartesian, self.B_cartesian, self.Q_cartesian, self.R_cartesian)[0]