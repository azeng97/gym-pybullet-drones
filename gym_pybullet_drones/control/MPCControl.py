import math
import numpy as np
import pybullet as p

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.envs.BaseAviary import DroneModel, BaseAviary
from gym_pybullet_drones.utils.utils import nnlsRPM

from casadi import *
import do_mpc


  

class MPCControl(BaseControl):
    """Generic PID control class without yaw control.

    Based on https://github.com/prfraanje/quadcopter_sim.

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
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
        self.make_model() 
        self.reset()

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
    
    def make_model(self):

      g = self.G
      m = self.M
      Ix = self.I_X
      Iy = self.I_Y
      Iz = self.I_Z

      print(self.M)

      model_type = 'continuous'
      model = do_mpc.model.Model(model_type)
      x = model.set_variable('_x', 'x')
      y = model.set_variable('_x', 'y')
      z = model.set_variable('_x', 'z')
      r = model.set_variable('_x', 'r')
      p = model.set_variable('_x', 'p')
      ya = model.set_variable('_x', 'ya')
      dx = model.set_variable('_x', 'dx')
      dy = model.set_variable('_x', 'dy')
      dz = model.set_variable('_x', 'dz')
      dr = model.set_variable('_x', 'dr')
      dp = model.set_variable('_x', 'dp')
      dya = model.set_variable('_x', 'dya')

      u1 = model.set_variable('_u', 'u1')
      u2 = model.set_variable('_u', 'u2')
      u3 = model.set_variable('_u', 'u3')
      u4 = model.set_variable('_u', 'u4')

      zRef = model.set_variable('_tvp', var_name='zRef')
      yaRef = model.set_variable('_tvp', var_name='yaRef')


      model.set_rhs('x', dx)
      model.set_rhs('y', dy)
      model.set_rhs('z', dz)
      model.set_rhs('r', dr)
      model.set_rhs('p', dp)
      model.set_rhs('ya', dya)
      model.set_rhs('dx', (np.cos(r) * np.sin(p) * np.cos(y) + np.sin(r) * np.sin(y)) * u1 / m)
      model.set_rhs('dy', (np.cos(r) * np.sin(p) * np.sin(y) - np.sin(r) * np.sin(y)) * u1 / m)
      model.set_rhs('dz', -g + np.cos(r) * np.cos(p) * u1 / m)
      #model.set_rhs('dz', -g + u1 / m)
      model.set_rhs('dr', dp * dy * (Ix - Iz) / Ix + u2 / Ix)
      model.set_rhs('dp', dr * dy * (Iz - Ix) / Iy + u3 / Iy)
      model.set_rhs('dya', dr * dp * (Ix - Iy) / Iz + u4 / Iz)

      model.setup()

      self.model = model

      self.mpc = do_mpc.controller.MPC(self.model)
      setup_mpc = {
          'n_horizon': 10,
          'n_robust': 0,
          'open_loop': 0,
          't_step': 0.1,
          'state_discretization': 'collocation',
          'collocation_type': 'radau',
          'collocation_deg': 3,
          'collocation_ni': 1,
          'store_full_solution': True,
          # Use MA27 linear solver in ipopt for faster calculations:
          'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}
      }
      self.mpc.set_param(**setup_mpc)

      mterm = 5*(z - zRef)**2 + 0.1*(ya - yaRef)**2 + 1e-5*dz**2 + 1e-10*(x**2 + y**2 + dx**2 + dy**2) 
      lterm = mterm

      self.mpc.set_objective(mterm=mterm, lterm=lterm)
      self.mpc.set_rterm(u1=5e-1, u2=5e-1, u3=5e-1, u4=5e-1)
      #self.mpc.set_rterm(u1=5e-2)

      self.mpc.bounds['lower', '_u', 'u1'] = 0
      self.mpc.bounds['upper', '_u', 'u1'] = self.MAX_THRUST

      self.mpc.bounds['lower', '_x', 'r'] = -np.pi
      self.mpc.bounds['upper', '_x', 'r'] = np.pi

      self.mpc.bounds['lower', '_x', 'p'] = -np.pi/2
      self.mpc.bounds['upper', '_x', 'p'] = np.pi/2

      self.mpc.bounds['lower', '_x', 'ya'] = -np.pi
      self.mpc.bounds['upper', '_x', 'ya'] = np.pi

      self.mpc.bounds['lower', '_u', 'u2'] = -self.MAX_XY_TORQUE
      self.mpc.bounds['upper', '_u', 'u2'] = self.MAX_XY_TORQUE

      self.mpc.bounds['lower', '_u', 'u3'] = -self.MAX_XY_TORQUE
      self.mpc.bounds['upper', '_u', 'u3'] = self.MAX_XY_TORQUE

      self.mpc.bounds['lower', '_u', 'u4'] = -self.MAX_Z_TORQUE
      self.mpc.bounds['upper', '_u', 'u4'] = self.MAX_Z_TORQUE

      tvp_template = self.mpc.get_tvp_template()
      def tvp_fun(t_now):
        # print("t_now")
        # print(t_now)
        for k in range(10+1):
          tvp_template['_tvp', k, 'zRef'] = 5
          tvp_template['_tvp', k, 'yaRef'] = np.pi/8

        return tvp_template
      self.mpc.set_tvp_fun(tvp_fun)

      self.mpc.setup()
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

        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        x0 = np.hstack((cur_pos, cur_rpy, cur_vel, cur_ang_vel))
        
        #x0 = np.hstack((cur_pos[2], cur_vel[2]))
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()
        u0 = self.mpc.make_step(x0)
        # print(x0)
        # print(u0)
        u_final = nnlsRPM(thrust=u0[0][0],
                       x_torque=u0[1][0],
                       y_torque=u0[2][0],
                       z_torque=u0[3][0],
                       counter=self.control_counter,
                       max_thrust=self.MAX_THRUST,
                       max_xy_torque=self.MAX_XY_TORQUE,
                       max_z_torque=self.MAX_Z_TORQUE,
                       a=self.A,
                       inv_a=self.INV_A,
                       b_coeff=self.B_COEFF,
                       gui=True
                       )
        # print(u_final)
        # sys.exit()
        return u_final

    ################################################################################
