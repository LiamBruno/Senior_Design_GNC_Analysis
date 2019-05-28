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
        self.kernel = SPK.open('de430.bsp') 

        # Constants for each relevant celestial body:
        self.EARTH = 3
        self.MOON = 301
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

    def getError(self, eps, eta, w, R, V, UTC, a):
        if self.mode == "Nadir":

            C_princ_inertial = QtoC(hstack([eps, eta]))
            h = cross(R, V)
            zhat = -R / norm(R)
            yhat = -h / norm(h)
            xhat = cross(yhat, zhat)
            C_LVLH_ECI = vstack([xhat, yhat, zhat])
            C_princ_LVLH = C_princ_inertial @ (C_LVLH_ECI.T)

            q_LVLH_princ = CtoQ(C_princ_LVLH)
            eps_e = q_LVLH_princ[0:3]
            
            w_c = C_princ_inertial @ (h/(norm(R)**2))
            w_e = w - w_c

            return eps_e, w_e

        elif self.mode == "Sun_Point":

            C_princ_inertial = QtoC(hstack([eps, eta]))

            # Ephemerides:
            R_sun, V_sun = self.kernel[self.CENTER, self.SUN].compute_and_differentiate(ut_to_jd(UTC))
            R_earth_moon, V_earth_moon = self.kernel[self.EARTH, self.MOON].compute_and_differentiate(ut_to_jd(UTC)) # Earth to Moon
            V_earth_moon /= 86400
            V_sun /= 86400 # [km/s]
            R_earth, V_earth = self.kernel[self.CENTER, self.EARTH].compute_and_differentiate(ut_to_jd(UTC)) # Center of Solar System to Earth
            V_earth /= 86400

            R_sc_sun = R_sun - R_earth - R_earth_moon - R
            V_sc_sun = V_earth + V_earth_moon + V - V_sun

            # Calculate quaternion error:
            a = self.C_princ_body @ a
            theta = angleBetween(a, C_princ_inertial@(-R_sc_sun))
            n_vec = cross(a, C_princ_inertial@(R_sc_sun))
            n_vec = n_vec/norm(n_vec)
            eps_e = sin(theta/2)*n_vec

            # Calculate angular velocity error:
            w_e = w - C_princ_inertial @ (cross(-R_sc_sun, V_sc_sun)/norm(R_sc_sun)**2)
            return eps_e, w_e

        elif self.mode == 'Earth_Point':

            C_princ_inertial = QtoC(hstack([eps, eta]))

            # Find spacecraft to Earth vector:
            R_earth_moon, V_earth_moon = self.kernel[self.EARTH, self.MOON].compute_and_differentiate(ut_to_jd(UTC)) # Earth to Moon
            V_earth_moon /= 86400 # [km/s]
            R_sc_earth = -R - R_earth_moon # Spacecraft to Earth
            V_sc_earth = V_earth_moon + V

            # Calculate quaternion error:
            a = self.C_princ_body @ a
            theta = angleBetween(a, C_princ_inertial@(-R_sc_earth))
            n_vec = cross(a, C_princ_inertial@(R_sc_earth))
            n_vec = n_vec/norm(n_vec)
            eps_e = sin(theta/2)*n_vec

            # Calculate angular velocity error:
            w_e = w - C_princ_inertial @ (cross(-R_sc_earth, V_sc_earth)/norm(R_sc_earth)**2)
            return eps_e, w_e
    #getError

    def command_torque(self, eps, eta, w, R, V, UTC, a):
        mu = 4.9048695e3 # [km^3/s^2]
        E_err, w_err = self.getError(eps, eta, w, R, V, UTC, a)
        C_princ_inertial_estimate = QtoC(hstack([eps, eta]))
        Rprinc = C_princ_inertial_estimate @ R
        Tgg = (3*mu/norm(R)**5)*(crux(Rprinc) @ (self.J @ Rprinc)) 
        Tc = -self.kp@E_err - self.kd@w_err - Tgg
        return Tc
    #command_torque

    def command_wheel_acceleration(self, Tc):

        wheel_torques = self.As_bar_inv@hstack([-Tc, 0])
        wheel_acceleration = inv(self.Ics)@wheel_torques

        return wheel_acceleration
    #command_wheel_rates

#controller