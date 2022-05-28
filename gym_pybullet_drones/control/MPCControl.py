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
      vx = model.set_variable('_x', 'vx')
      vy = model.set_variable('_x', 'vy')
      vz = model.set_variable('_x', 'vz')
      phi = model.set_variable('_x', 'phi')
      theta = model.set_variable('_x', 'theta')
      psi = model.set_variable('_x', 'psi')
      p = model.set_variable('_x', 'p')
      q = model.set_variable('_x', 'q')
      r = model.set_variable('_x', 'r')

      u1 = model.set_variable('_u', 'u1')
      u2 = model.set_variable('_u', 'u2')
      u3 = model.set_variable('_u', 'u3')
      u4 = model.set_variable('_u', 'u4')

      zRef = model.set_variable('_tvp', var_name='zRef')
      yaRef = model.set_variable('_tvp', var_name='yaRef')

      R_BI = np.zeros((3, 3))
      R_BI[0, 0] = np.cos(theta)*np.cos(psi)
      R_BI[0, 1] = np.cos(theta)*np.sin(psi)
      R_BI[0, 2] = -np.sin(theta)

      R_BI[1, 0] = np.sin(theta)*np.cos(psi)*np.sin(phi) - np.sin(psi)*np.cos(phi)
      R_BI[1, 1] = np.sin(theta)*np.sin(psi)*np.sin(phi) + np.cos(psi)*np.cos(phi)
      R_BI[1, 2] = np.cos(theta)*np.sin(phi)

      R_BI[2, 0] = np.sin(theta)*np.cos(psi)*np.cos(phi) + np.sin(psi)*np.sin(phi)
      R_BI[2, 1] = np.sin(theta)*np.sin(psi)*np.cos(phi) - np.cos(psi)*np.sin(phi)
      R_BI[2, 2] = np.cos(theta)*np.cos(phi)

      W = np.zeros((3,3))
      W[0,0] = 1
      W[0,1] = np.sin(phi)*np.tan(theta)
      W[0,2] = np.cos(phi)*np.tan(theta)

      W[1,0] = 0
      W[1,1] = np.cos(phi)
      W[1,2] = -np.sin(phi)

      W[2,0] = 0
      W[2,1] = np.sin(phi)*np.arccos(theta)
      W[2,2] = np.cos(theta)*np.arccos(theta)

      J = np.zeros((3,3))
      J[0,0] = Ix 
      J[1,1] = Iy 
      J[2,2] = Iz

      e3 = np.zeros((3,1))
      e3[2,0] = 1

      Vb = np.array([vx, vy, vz]).T
      OMEGA = np.array([p, q, r]).T
      TAU = np.array([u2, u3, u4]).T

      dyn = np.vstack((np.dot(R_BI.T, Vb),
                      -np.cross(OMEGA, Vb) + np.dot(R_BI, -g*e3) + u1 / m * e3,
                      np.dot(W,OMEGA),
                      np.dot(np.linalg.inv(J), -np.cross(OMEGA, np.dot(J, OMEGA)) + TAU))) 


      model.set_rhs('x', dyn[0])
      model.set_rhs('y', dyn[1])
      model.set_rhs('z', dyn[2])
      model.set_rhs('vx', dyn[3])
      model.set_rhs('vy', dyn[4])
      model.set_rhs('vz', dyn[5])
      model.set_rhs('phi', dyn[6])
      model.set_rhs('theta', dyn[7])
      model.set_rhs('psi', dyn[8])
      #model.set_rhs('dz', -g + u1 / m)
      model.set_rhs('p', dyn[9])
      model.set_rhs('q', dyn[10])
      model.set_rhs('r', dyn[11])

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

      mterm = 5*(z - zRef)**2 + 0.1*(theta - yaRef)**2 + 1e-5*dz**2 + 1e-10*(x**2 + y**2 + vx**2 + vy**2) 
      lterm = mterm

      self.mpc.set_objective(mterm=mterm, lterm=lterm)
      self.mpc.set_rterm(u1=5e-1, u2=5e-1, u3=5e-1, u4=5e-1)
      #self.mpc.set_rterm(u1=5e-2)

      self.mpc.bounds['lower', '_u', 'u1'] = 0
      self.mpc.bounds['upper', '_u', 'u1'] = self.MAX_THRUST

      self.mpc.bounds['lower', '_x', 'phi'] = -np.pi
      self.mpc.bounds['upper', '_x', 'phi'] = np.pi

      self.mpc.bounds['lower', '_x', 'theta'] = -np.pi/2
      self.mpc.bounds['upper', '_x', 'theta'] = np.pi/2

      self.mpc.bounds['lower', '_x', 'psi'] = -np.pi
      self.mpc.bounds['upper', '_x', 'psi'] = np.pi

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
          tvp_template['_tvp', k, 'yaRef'] = 0#np.pi/8

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
