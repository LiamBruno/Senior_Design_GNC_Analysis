from numpy import *
from numpy.linalg import inv

class Controller:

	def __init__(self, sc_inertia, wheel_inertias, As, wheel_rates = array([0,0,0,0]), mode = "Nadir"):
		self.As = As #assumes 4 wheels
		self.As_bar_inv = inv(vstack([self.As, array([1, -1, 1, -1])]))
		self.J = sc_inertia
		self.Ics = diag(wheel_inertias)
		self.kp = 1
		self.kd = 1
		self.wheel_rates = wheel_rates
		self.mode = mode
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

	def getIcs(self):
		return self.Ics
	#getIcs

	def set_gains(self, kp, kd):
		self.kp = kp
		self.kd = kd
	#set_gains

	def set_estimate(self, EKF_output):
		self.estimate = EKF_output
	#set_estimate

	def get_estimate(self):
		return self.estimate
	#get_estimate

	def getError(self, eps, eta, w, R = None, V = None, UTC = None):
		if self.mode == "Nadir":
			C_bI = QtoC(hstack([eps, eta]))
			h = cross(R, V)
			zhat = -R / norm(R)
			yhat = -h / norm(h)
			xhat = cross(yhat, zhat)
			C_LVLH_ECI = vstack([xhat, yhat, zhat])
			C_b_LVLH = C_bI @ (C_LVLH_ECI.transpose())
			eps_e = CtoQ(C_b_LVLH)[0:3]
			w_c = C_bI @ (h / (norm(R) ** 2))
			w_e = w - w_c
			return eps_e, w_e
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