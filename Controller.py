from numpy import *
from numpy.linalg import inv
from AERO_TOOLKIT_NEW import *
import pyquaternion
from jplephem.spk import SPK

class Controller:

	def __init__(self, sc_inertia, wheel_inertias, As, C_principal_body, wheel_rates = array([0,0,0,0]), mode = "Nadir"):
		self.As = As #assumes 4 wheels
		self.As_bar_inv = inv(vstack([self.As, array([1, -1, 1, -1])]))
		self.J = sc_inertia*1.05
		self.Ics = wheel_inertias
		self.kp = 1
		self.kd = 1
		self.wheel_rates = wheel_rates
		self.mode = mode
		self.C_princ_body = C_principal_body

		# Load .bsp containing JPL ephemerides:
		self.kernel = SPK.open('de430.bsp') # Load .bsp file containing ephemerides

		# Constants for each relevant celestial body:
		self.EARTH = 3
		self.MOON = 399
		self.SUN = 10
		self.CENTER = 0
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

	def getError(self, eps, eta, w, R = None, V = None, UTC = None, a = None):
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

			theta = arccos(eta_e)*2

			return eps_e, w_e, theta

		elif self.mode == "Sun_Point":
			C_princ_inertial = QtoC(hstack([eps, eta]))

			# Ephemerides:
			R_sun, V_sun = self.kernel[0, self.SUN].compute_and_differentiate(ut_to_jd(UTC))
			V_sun /= 86400 # [km/s]
			R_earth, V_earth = self.kernel[0, self.EARTH].compute_and_differentiate(ut_to_jd(UTC))
			R_earth_moon, V_earth_moon = self.kernel[3, self.MOON].compute_and_differentiate(ut_to_jd(UTC))
			R_moon = R_earth + R_earth_moon
			V_moon = V_earth + V_earth_moon
			V_moon /= 86400 # [km/s]
			R_sc = R_moon + R # Inertial position of S/C relative to solar system barycetner [km]
			V_sc = V_moon + V # Inertial velocity of S/C relative to solar system barycetner [km/s]


			# Calculate quaternion error:
			a_inertial = C_princ_inertial.T@self.C_princ_body@a
			theta = angleBetween(a_inertial, -R_sc)
			eps_e = cross(a_inertial, -R_sc)/norm(cross(a_inertial, R_sc))*sin(theta/2)

			# Calculate angular velocity error:
			w_e = w - C_princ_inertial @ (cross(R_sc, V_sc)/norm(R_sc)**2)
			return eps_e, w_e, theta
		#elif
	#getError

	def command_torque(self, eps, eta, w, R = None, V = None, UTC = None, a = None):
		E_err, w_err, theta_err = self.getError(eps, eta, w, R, V, UTC, a)
		Tc = -self.kp@E_err -self.kd@w_err
		return Tc, theta_err
	#command_torque

	def command_wheel_acceleration(self, Tc):
		wheel_torques = self.As_bar_inv@hstack([-Tc, 0])
		wheel_acceleration = inv(self.Ics)@wheel_torques
		return wheel_acceleration
	#command_wheel_rates

#controller