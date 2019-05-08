from numpy import *
from numpy.linalg import inv
from AERO_TOOLKIT_NEW import *
import pyquaternion

class Controller:

	def __init__(self, sc_inertia, wheel_inertias, As, C_principal_body, wheel_rates = array([0,0,0,0]), mode = "Nadir"):
		self.As = As #assumes 4 wheels
		self.As_bar_inv = inv(vstack([self.As, array([1, -1, 1, -1])]))
		self.J = sc_inertia
		self.Ics = wheel_inertias
		self.kp = 1
		self.kd = 1
		self.wheel_rates = wheel_rates
		self.mode = mode
		self.C_princ_body = C_principal_body
	#constructor

	def calc_gains(self, damping_ratio, settling_time):
		w_n = 4.4/(damping_ratio*settling_time)
		kp = 2*(w_n**2)*self.J
		kd = 2*damping_ratio*w_n*self.J
		return kp, kd
	#calc_gains

	def set_mode(self, mode):
		self.mode = mode
	#set_mode

	def set_gains(self, kp, kd):
		self.kp = kp
		self.kd = kd
	#set_gains

	def getError(self, eps, eta, w, R = None, V = None, UTC = None):
		if self.mode == "Nadir":
			C_princ_inertial = QtoC(hstack([eps, eta]))
			h = cross(R, V)
			zhat = -R / norm(R)
			yhat = -h / norm(h)
			xhat = cross(yhat, zhat)
			C_LVLH_ECI = vstack([xhat, yhat, zhat])
			C_princ_LVLH = C_princ_inertial @ (C_LVLH_ECI.T)

			q_LVLH_princ = CtoQ(C_princ_LVLH)
			eps_LVLH_princ = q_LVLH_princ[0:3]
			eta_LVLH_princ = q_LVLH_princ[3]

			q_c = CtoQ(self.C_princ_body)
			eps_e, eta_e = quat_mult(-q_c[0:3], q_c[3], eps_LVLH_princ, eta_LVLH_princ)

			w_c = C_princ_inertial @ (h / (norm(R) ** 2))
			w_e = w - w_c
			return eps_e, w_e

		if self.mode == "Tumble":
			return zeros((3,)), zeros((3,))
		#if
	#getError

	def command_torque(self, eps, eta, w, R = None, V = None, UTC = None):
		E_err, w_err = self.getError(eps, eta, w, R, V, UTC)
		Tc = -self.kp@E_err -self.kd@w_err
		return Tc
	#command_torque

	def command_wheel_torques(self, Tc):
		wheel_torques = self.As_bar_inv@hstack([Tc, 0])
		wheel_acceleration = inv(self.Ics)@wheel_torques
		return wheel_acceleration
	#command_wheel_rates

#controller