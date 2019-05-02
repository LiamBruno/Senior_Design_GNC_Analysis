from numpy import *
import matplotlib.pyplot as plt


class Controller():

	def __init__(self, sc_inertia, wheel_tilt_angle, wheel_inertias, wheel_rates = None):
		#assumes 4 wheels

		s = sin(wheel_tilt_angle)
		c = cos(wheel_tilt_angle)
		self.As = array([[s, 0, -s, 0],
						[0, s, 0, -s],
						[c, c, c, c]])
		self.As_bar_inv = inv(vstack([self.As, array([1, -1, 1, -1])]))

		self.J = sc_inertia
		self.Ics = diag(wheel_inertias)
		self.kp = 1
		self.kd = 1
		if wheel_rates == None:
			self.wheel_rates = array([0,0,0,0])
		else:
			self.wheel_rates = wheel_rates

		self.E_target = array([0,0,0])
		self.n_target = 1
		self.w_target = array([0,0,0])

	def calc_gains(self, damping_ratio, angular_frequency):
		kp = 2*self.J*angular_frequency**2
		kd = 2*J*damping_ratio*angular_frequency
		return kp, kd

	def set_gains(self, kp, kd):
		self.kp = kp
		self.kd = kd

	def set_target(E_target, n_target, w_target, clocking = False):
		self.E_target = E_target
		self.n_target = n_target
		self.w_target = w_target


	def command_torque(self, E, n, w):
		E_err = E
		w_err = w
		Tc = -kp@E_err -kd@w_err
		return Tc


	def command_wheel_rates(self, Tc):
		wheel_torques = self.As_bar_inv@hstack([Tc, 0])
		wheel_acceleration = inv(self.Ics)@wheel_torques
		return wheel_acceleration
