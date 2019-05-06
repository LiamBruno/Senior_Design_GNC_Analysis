from StarTrackerFilter import *
from AERO_TOOLKIT_NEW import *
from pyquaternion import *
import matplotlib.pyplot as plt
from Controller import *

def main():

    # Initial state:
    n = 0.5
    E = array([0.5, 0.5, 0.5])
    w = array([0, 0, 0])
    w_wheels = zeros(4)
    mu = 4.9048695e3 # [km^3/s^2]
    Rm = 1731
    R = array([Rm + 100, 0, 0]) # [km]
    V = array([0, sqrt(mu/norm(R)), 0])
    T = 2*pi*sqrt(norm(R)**3/mu)

    # SC Properties:
    I = diag(array([5000, 12000, 13000])) # Inertia matrix in principal body frame [kg*m^2]

    # Wheel properties:
    m = 50 # [kg]
    r = .3 # [m]
    h = .1 # [m]
    Iw = (1/2)*m*r**2 # Inertia of wheels around spin axis [kg*m^2]

    # Noise estimates (standard deviations) for EKF:
    PROCESS_NOISE = sqrt(1e-12)
    MEASUREMENT_NOISE = sqrt(1e-11)
    COVARIANCE_GUESS = sqrt(1e-9)
    STAR_TRACKER_NOISE = sqrt(1e-12)
    GYRO_NOISE = sqrt(1e-10)

    # Initial EKF estimate:
    aguess = array([1, 5, -3])/norm(array([1, 5, -3]))
    thetaguess = 3
    Eguess = aguess*sin(thetaguess/2)
    nguess = cos(thetaguess/2)
    w_guess = array([0.05, 0.05, -0.05])

    # Instantiate EKF object:
    EKF = StarTracker_Filter(Eguess, nguess, w_guess, I, Iw, PROCESS_NOISE, MEASUREMENT_NOISE, COVARIANCE_GUESS)

    # Actuator Properties and Configuration:
    WHEEL_TILT = 35*(pi/180) # [rad]
    WHEEL_INERTIAS = [Iw]*4 # [kg*m^2]
    DAMPING_RATIO = .70
    SETTLING_TIME =  5*60# [sec]
    E_TARGET = array([0,0,0])
    N_TARGET = 1
    W_TARGET = array([0,0,0])
    s = sin(WHEEL_TILT)
    c = cos(WHEEL_TILT)
    AS = array([[s, 0, -s, 0],
                [0, s, 0, -s],
                [c, c, c,  c]])

    # Instantiate Controller Object:
    controller = Controller(I, WHEEL_INERTIAS, EKF.getState(), AS)
    KP, KD, = controller.calc_gains(DAMPING_RATIO, SETTLING_TIME)
    controller.set_gains(KP, KD)
    controller.set_target(E_TARGET, N_TARGET, W_TARGET) # Sets desired profile

    state = hstack([E, n, w, w_wheels, R, V])

    dt = .5
    solver = ode(propagateTruth)
    solver.set_integrator('lsoda', max_step = dt, atol = 1e-8, rtol = 1e-8)
    solver.set_initial_value(state, 0)
    I_cs = controller.getIcs()
    solver.set_f_params(I, I_cs, AS, mu, zeros(3), zeros(4))
    
    tspan = T/5 # Total simulation time [sec]
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
        controller.set_estimate(estimate)
        Tc = controller.command_torque()
        wheel_accel = controller.command_wheel_torques(Tc)
        solver.set_f_params(I, I_cs, AS, mu, Tc, wheel_accel)

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

    plt.semilogy(t/T, 3600*(180/pi)*pointing_error, '.')
    plt.grid()
    plt.title('Pointing Knowledge Error [Degrees]')
    plt.xlabel('Time [Number of Orbits]')

    fig4 = plt.figure()

    w1 = newstate[:, 4]
    w2 = newstate[:, 5]
    w3 = newstate[:, 6]

    plt.plot(t / T, (180 / pi) * w1)
    plt.plot(t / T, (180 / pi) * w2)
    plt.plot(t / T, (180 / pi) * w3)
    plt.grid()
    plt.title('Body Rates [deg/s]')
    plt.xlabel('Time [Number of Orbits]')
    plt.legend(['Body X', 'Body Y', 'Body Z'])

    fig5 = plt.figure()

    plt.plot(t / T, Iw*newstate[:, 7])
    plt.plot(t / T, Iw*newstate[:, 8])
    plt.plot(t / T, Iw*newstate[:, 9])
    plt.plot(t / T, Iw*newstate[:, 10])
    plt.grid()
    plt.title('Wheel Momentum Storage')
    plt.xlabel('Time [Number of Orbits]')
    plt.ylabel('Angular Momentum [Nms]')
    plt.legend(['1', '2', '3', '4'])

    plt.show()
#main
    
def propagateTruth(t, state, I, Ics, As, mu, Tc, wheel_accel):
    E = state[0:3]
    n = state[3]
    w = state[4:7]
    w_wheels = state[7:11]
    R = state[11:14]
    V = state[14:]

    dR = V
    dV = -(mu/norm(R)**3)*R

    dE = .5*(n*identity(3) + crux(E))@w
    dn = -.5*dot(E, w)

    C_bI = QtoC(hstack([E, n]))
    Rb = C_bI @ R
    Tgg = (3*mu/norm(R)**5)*(crux(Rb) @ (I @ Rb))
    h_w = As @ (Ics @ w_wheels)
    dw = inv(I) @ (Tgg + Tc - crux(w) @ (I@w + h_w))

    wwdot = wheel_accel
    
    return hstack([dE, dn, dw, wwdot, dR, dV])
#propagateTruth

if __name__ == "__main__":
    main()
#if



