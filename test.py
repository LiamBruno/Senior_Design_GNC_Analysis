from StarTrackerFilter import *
from AERO_TOOLKIT_NEW import *
from pyquaternion import *
import matplotlib.pyplot as plt
from Controller import *

def main():

    n = 0.5
    E = array([0.5, 0.5, 0.5])
    w = array([.01, 0.07, -0.08])
    w_wheels = zeros(4)

    mu = 4.9048695e3 # [km^3/s^2]
    Rm = 1731
    R = array([Rm + 100, 0, 0]) # [km]
    V = array([0, sqrt(mu/norm(R)), 0])
    T = 2*pi*sqrt(norm(R)**3/mu)

    # SC Properties:
    L = 3 # [m]
    W = 3 # [m]
    H = 5 # [m]

    m = 500 # [kg]

    I = array([[(1/12)*m*(L**2 + W**2), 0, 0],[0, (1/12)*m*(L**2 + H**2), 0],[0, 0, (1/12)*m*(H**2 + W**2)]])
    Iw = 0.5 # Inertia of wheels around spin axis [kg*m^2]
    wheel_cant = pi/8 # [rad]

    # Noise estimates (standard deviations) for EKF:
    PROCESS_NOISE = sqrt(1e-12)
    MEASUREMENT_NOISE = sqrt(1e-12)
    COVARIANCE_GUESS = sqrt(1e-10)
    STAR_TRACKER_NOISE = sqrt(1e-12)
    GYRO_NOISE = sqrt(1e-12)

    aguess = array([1, 5, -3])/norm(array([1,5,-3]))
    thetaguess = 3
    Eguess = aguess*sin(thetaguess/2)
    nguess = cos(thetaguess/2)
    w_guess = array([0.05, 0.05, -0.05])

    EKF = StarTracker_Filter(Eguess, nguess, w_guess,  I, Iw, wheel_cant, PROCESS_NOISE, MEASUREMENT_NOISE, COVARIANCE_GUESS)

    WHEEL_TILT = 35/180*pi #35 degrees but in radians
    WHEEL_INERTIAS = [Iw]*4
    DAMPING_RATIO = .65
    SETTLING_TIME = 30 # [sec]
    E_TARGET = array([0,0,0])
    N_TARGET = 1
    W_TARGET = array([0,0,0])
    controller = Controller(I, WHEEL_TILT, WHEEL_INERTIAS)
    KP, KD, = controller.calc_gains(DAMPING_RATIO, SETTLING_TIME)
    controller.set_gains(KP, KD)
    controller.set_target(E_TARGET, N_TARGET, W_TARGET)

    state = hstack([E, n, w, w_wheels, R, V])
    
    solver = ode(propagateTruth)
    solver.set_integrator('dopri5')
    solver.set_initial_value(state, 0)
    solver.set_f_params(I, mu)
    
    tspan = T/10 # Total simulation time [sec]
    dt = 0.5
    t = [] # [sec]
    newstate = []
    measurements = []
    state_estimate = []
    ang_vel_error = []
    q_error = []

    newstate.append(solver.y)
    t.append(solver.t)
    state_estimate.append(EKF.getState())
    q_true = Quaternion(array = solver.y[0:4])
    q_estimate = Quaternion(array = EKF.getState()[0:4])
    q_e = q_true.conjugate*q_estimate
    q_error.append(array([q_e[1], q_e[2], q_e[3], q_e[0]]))
    pointing_error = []

    # Pointing error calc:
    C_bI_estimate = QtoC(EKF.getState()[0:4])
    z_I_estimate = C_bI_estimate[2, :]
    C_bI_true = QtoC(solver.y[0:4])
    z_I_true = C_bI_true[2, :]
    pointing_error.append(angleBetween(z_I_true, z_I_estimate))

    percentage = 10
    while solver.successful() and (solver.t < tspan):

        # Integrate:
        solver.integrate(solver.t + dt)
        t.append(solver.t)
        newstate.append(solver.y)

        # Simulate measurements:
        q_true = Quaternion(array = array([solver.y[3], solver.y[0], solver.y[1], solver.y[2]]))
        dq = Quaternion(array = hstack([1, random.normal(0, STAR_TRACKER_NOISE, (3,))/2]))
        q_measurement = dq * q_true
        q_measurement = array([q_measurement[1], q_measurement[2], q_measurement[3], q_measurement[0]])
        w_measurement = solver.y[4:7] + random.normal(0, GYRO_NOISE, (3,))

        # Update the filter:
        measurement = hstack([q_measurement, w_measurement])
        measurements.append(measurement)
        estimate = EKF.update(measurement, dt)
        ang_vel_error.append(norm(estimate[4:7]) - norm(solver.y[4:7]))
        state_estimate.append(estimate)

        # Pointing error calc:
        C_bI_estimate = QtoC(estimate[0:4])
        z_I_estimate = C_bI_estimate[2, :]
        C_bI_true = QtoC(solver.y[0:4])
        z_I_true = C_bI_true[2, :]
        pointing_error.append(angleBetween(z_I_true, z_I_estimate))

        # Quaternion error calc:
        q_estimate = Quaternion(array = array([estimate[3], estimate[0], estimate[1], estimate[2]]))
        q_e = q_true.conjugate*q_estimate
        q_error.append(array([q_e[1], q_e[2], q_e[3], q_e[0]]))
        completion = solver.t/tspan*100

        # Progress:
        if completion > percentage:
            print(percentage,"percent complete")
            percentage+= 10
        #if
    #while

    t = hstack(t)
    newstate = vstack(newstate)
    state_estimate = vstack(state_estimate)
    q_error = vstack(q_error)
    pointing_error = hstack(pointing_error)

    fig, axes1 = plt.subplots(4, 1, squeeze = False)

    axes1[0][0].plot(t/T, state_estimate[:,0], '.')
    axes1[0][0].plot(t/T, newstate[:,0])
    axes1[0][0].set_title('eps1')
    axes1[0][0].legend(['Truth', 'Estimate'])

    axes1[1][0].plot(t/T, state_estimate[:,1], '.')
    axes1[1][0].plot(t/T, newstate[:,1])
    axes1[1][0].set_title('eps2')

    axes1[2][0].plot(t/T, state_estimate[:,2], '.')
    axes1[2][0].plot(t/T, newstate[:,2])
    axes1[2][0].set_title('eps3')

    axes1[3][0].plot(t/T, state_estimate[:,3], '.')
    axes1[3][0].plot(t/T, newstate[:,3])
    axes1[3][0].set_title('eta')

    fig1 = plt.figure()

    plt.plot(t/T, q_error[:, 0])
    plt.plot(t/T, q_error[:, 1])
    plt.plot(t/T, q_error[:, 2])
    plt.plot(t/T, q_error[:, 3])
    plt.grid()
    plt.title('Quaternion Error')
    plt.xlabel('Time [Number of Orbits]')
    plt.legend(['q_i', 'q_j', 'q_k', 'q_r'])

    fig2 = plt.figure()

    w1_error = newstate[:, 4] - state_estimate[:,4]
    w2_error = newstate[:, 5] - state_estimate[:,5]
    w3_error = newstate[:, 6] - state_estimate[:,6]

    plt.plot(t/T, (180/pi)*w1_error)
    plt.plot(t/T, (180/pi)*w2_error)
    plt.plot(t/T, (180/pi)*w3_error)
    plt.grid()
    plt.title('Angular Velocity Error [deg/s]')
    plt.xlabel('Time [Number of Orbits]')
    plt.legend(['Body X', 'Body Y', 'Body Z'])

    fig3 = plt.figure()

    plt.plot(t/T, (180/pi)*pointing_error, '.')
    plt.grid()
    plt.title('Pointing Knowledge Error [Degrees]')
    plt.xlabel('Time [Number of Orbits]')

    plt.show()
#main
    
def propagateTruth(t, state, I, mu):
    E = state[0:3]
    n = state[3]
    w = state[4:7]
    R = state[10:13]
    V = state[13:16]

    dR = V
    dV = -(mu/norm(R)**3)*R

    dE = .5*(n*identity(3) + crux(E))@w
    dn = -.5*dot(E, w)

    C_bI = QtoC(hstack([E, n]))
    Rb = C_bI @ R
    Tgg = (3*mu/norm(R)**5)*(crux(Rb) @ (I @ Rb))

    dw = inv(I) @ (Tgg - crux(w)@ (I@w))
    
    wwdot = zeros(4)
    
    return hstack([dE, dn, dw, wwdot, dR, dV])
#propagateTruth

if __name__ == "__main__":
    main()
#if



