from scipy.integrate import ode
from AERO_TOOLKIT_NEW import *

class StarTracker_Filter:

    def __init__(self, E_initial, n_initial , w_initial, I_sc, Iwheels, PROCESS_NOISE, MEASUREMENT_NOISE, COVARIANCE_GUESS):
        self.I_sc = 1.05*I_sc
        self.Iwheels = Iwheels
        self.H = identity(7)
        self.state = hstack([E_initial, n_initial, w_initial])
        self.P = COVARIANCE_GUESS
        self.Q = PROCESS_NOISE
        self.R = MEASUREMENT_NOISE
        solver = ode(propagateEKF)
        solver.set_integrator('lsoda')
        solver.set_initial_value(self.state, 0)
        solver.set_f_params(self.I_sc)
        self.integrator = solver
    #constructor

    def calcPhi(self, dt):
        Ixx = self.I_sc[0][0]
        Iyy = self.I_sc[1][1]
        Izz = self.I_sc[2][2]

        eps1 = self.state[0]
        eps2 = self.state[1]
        eps3 = self.state[2]
        eta = self.state[3]
        w1 = self.state[4]
        w2 = self.state[5]
        w3 = self.state[6]

        F = array([[0,w3/2,-w2/2,w1/2,eta/2,-eps3/2,eps2/2],
                   [-w3/2,0,w1/2,w2/2,eps3/2,eta/2,-eps1/2],
                   [w2/2,-w1/2,0,w3/2,-eps2/2,eps1/2,eta/2],
                   [-w1/2,-w2/2,-w3/2,0,-eps1/2,-eps2/2,-eps3/2],
                   [0,0,0,0,0,(w3*(Iyy-Izz))/Ixx,(w2*(Iyy-Izz))/Ixx],
                   [0,0,0,0,-(w3*(Ixx-Izz))/Iyy,0,-(w1*(Ixx-Izz))/Iyy],
                   [0,0,0,0,(w2*(Ixx-Iyy))/Izz,(w1*(Ixx-Iyy))/Izz,0]])

        return F*dt + identity(7)
    #getF

    def predict_with_ode(self, dt):
        self.integrator.integrate(self.integrator.t + dt)
        return self.integrator.y
    #predict_with_ode

    def getdstate(self, predicted_state, Rprinc, Tc):
        mu = 4.9048695e3 # [km^3/s^2]
        E = predicted_state[0:3]
        n = predicted_state[3]
        w = predicted_state[4:7]
        dE = .5 * (n * identity(3) + crux(E)) @ w
        dn = -.5 * dot(E, w)
        Tgg = (3*mu/norm(Rprinc)**5)*(crux(Rprinc) @ (self.I_sc @ Rprinc))
        dw = inv(self.I_sc) @ (Tc + Tgg - crux(w) @ (self.I_sc @ w))
        return hstack([dE, dn, dw])
    #getdstate

    def predict_euler(self, dt, Rprinc, Tc):
        predicted_state = self.state
        steps = 5
        for i in range(steps):
            dstate = self.getdstate(predicted_state, Rprinc, Tc)
            predicted_state += dstate*(dt/steps)
        #for
        return predicted_state
    #predict_euler

    def update(self, z, dt, Rprinc, Tc):
        Phi = self.calcPhi(dt)
        x_prediction = self.predict_euler(dt, Rprinc, Tc)
        P_prediction = Phi@self.P@Phi.T + self.Q
        y = z - self.H@x_prediction
        S = self.R + self.H@self.P@self.H.T
        K = P_prediction@self.H.T@inv(S)
        self.state = x_prediction + K@y
        self.P = (identity(7) - K@self.H)@P_prediction
        return self.state
    #update

    def getState(self):
        return self.state
    #getState
# starTrackerFilter
