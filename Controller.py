from numpy import *
from numpy.linalg import inv

class Controller:

	def __init__(self, sc_inertia, wheel_inertias, current_estimate, As, wheel_rates = array([0,0,0,0])):

		#assumes 4 wheels

		self.As = As
		self.As_bar_inv = inv(vstack([self.As, array([1, -1, 1, -1])]))

		self.J = sc_inertia
		self.Ics = diag(wheel_inertias)
		self.kp = 1
		self.kd = 1
		self.estimate = current_estimate


		self.wheel_rates = wheel_rates

		self.E_target = array([0,0,0])
		self.n_target = 1
		self.w_target = array([0,0,0])
	#constructor

	def calc_gains(self, damping_ratio, settling_time):
		w_n = 4.4/(damping_ratio*settling_time)
		kp = 2*(w_n**2)*self.J
		kd = 2*damping_ratio*w_n*self.J
		return kp, kd
	#calc_gains

	def getIcs(self):
		return self.Ics
	#getIcs

	def set_gains(self, kp, kd):
		self.kp = kp
		self.kd = kd
	#set_gains

	def set_target(self, E_target, n_target, w_target, clocking = False):
		self.E_target = E_target
		self.n_target = n_target
		self.w_target = w_target
	#set_target

	def set_estimate(self, EKF_output):
		self.estimate = EKF_output
	#set_estimate

	def get_estimate(self):
		return self.estimate
	#get_estimate

	def command_torque(self):
		# THESE ERRORS NEED TO BE COMPUTED BY A SEPARATE GUIDANCE FUNCTION DEPENDING ON THE MODE:
		E_err = self.estimate[0:3]
		w_err = self.estimate[4:7]
		Tc = -self.kp@E_err -self.kd@w_err
		return Tc
	#command_torque

	def command_wheel_torques(self, Tc):
		wheel_torques = self.As_bar_inv@hstack([Tc, 0])
		wheel_acceleration = inv(self.Ics)@wheel_torques
		return wheel_acceleration
	#command_wheel_rates

#controller