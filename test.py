from StarTrackerFilter import *
from AERO_TOOLKIT_NEW import *
import matplotlib.pyplot as plt

def main():

    n = 1
    E = array([0, 0, 0])
    w = array([.1, 0.07, -0.04])
    w_wheels = zeros(4)

    # SC Properties:
    L = 3 # [m]
    W = 3 # [m]
    H = 5 # [m]

    m = 500 # [kg]

    I = array([[(1/12)*m*(L**2 + W**2), 0, 0],[0, (1/12)*m*(L**2 + H**2), 0],[0, 0, (1/12)*m*(H**2 + W**2)]])
    Iw = 0.5 # Inertia of wheels around spin axis [kg*m^2]
    wheel_cant = pi/8 # [rad]

    # Noise estimates (standard deviations) for EKF:
    PROCESS_NOISE = sqrt(1e-10)
    MEASUREMENT_NOISE = sqrt(1e-3)
    COVARIANCE_GUESS = sqrt(1e-3)
    STAR_TRACKER_NOISE = sqrt(1e-3)
    GYRO_NOISE = sqrt(1e-3)

    aguess = array([1,5,-3])/norm(array([1,5,-3]))
    thetaguess = 3
    Eguess = aguess*sin(thetaguess/2)
    nguess = cos(thetaguess/2)
    w_guess = array([0.5, 0.5, 0.5])

    EKF = StarTracker_Filter(Eguess, nguess, w_guess,  I, Iw, wheel_cant, PROCESS_NOISE, MEASUREMENT_NOISE, COVARIANCE_GUESS)
    state = hstack([E, n, w, w_wheels])
    
    solver = ode(propagateTruth)
    solver.set_integrator('dopri5')
    solver.set_initial_value(state, 0)
    solver.set_f_params(I)
    
    tspan = 1000 # Total simulation time [sec]
    dt = .1
    t = [] # [sec]
    newstate = []
    measurements = []
    state_estimate = []
    ang_vel_error = []

    newstate.append(solver.y)
    t.append(solver.t)
    state_estimate.append(EKF.getState())

    percentage = 10
    while solver.successful() and (solver.t < tspan):
        q_measurement = solver.y[0:4] + random.normal(0, STAR_TRACKER_NOISE, (4,))
        q_measurement = q_measurement/norm(q_measurement)
        w_measurement = solver.y[4:7] + random.normal(0, GYRO_NOISE, (3,))
        measurement = hstack([q_measurement, w_measurement])
        measurements.append(measurement)
        estimate = EKF.update(measurement, dt)
        ang_vel_error.append(norm(estimate[4:7]) - norm(solver.y[4:7]))
        state_estimate.append(estimate)
        solver.integrate(solver.t + dt)
        t.append(solver.t)
        newstate.append(solver.y)

        completion = solver.t/tspan*100
        if completion > percentage:
            print(percentage,"percent complete")
            percentage+= 10

    #while

    t = hstack(t)
    newstate = vstack(newstate)
    state_estimate = vstack(state_estimate)

    fig, axes1 = plt.subplots(4, 1, squeeze = False)

    axes1[0][0].plot(t, state_estimate[:,0], '.')
    axes1[0][0].plot(t, newstate[:,0])
    axes1[0][0].set_title('eps1')
    axes1[0][0].legend(['Truth', 'Estimate'])

    axes1[1][0].plot(t, state_estimate[:,1], '.')
    axes1[1][0].plot(t, newstate[:,1])
    axes1[1][0].set_title('eps2')

    axes1[2][0].plot(t, state_estimate[:,2], '.')
    axes1[2][0].plot(t, newstate[:,2])
    axes1[2][0].set_title('eps3')

    axes1[3][0].plot(t, state_estimate[:,3], '.')
    axes1[3][0].plot(t, newstate[:,3])
    axes1[3][0].set_title('eta')

    fig = plt.figure()

    w1_error = state_estimate[:,4] - newstate[:, 4]
    w2_error = state_estimate[:, 5] - newstate[:, 5]
    w3_error = state_estimate[:, 6] - newstate[:, 6]

    plt.plot(t, w1_error)
    plt.plot(t, w2_error)
    plt.plot(t, w3_error)
    plt.grid()
    plt.title('Angular Velocity Error [rad/s]')
    plt.legend(['Body X', 'Body Y', 'Body Z'])

    plt.show()
#main
    
def propagateTruth(t, state, I):
    E = state[0:3]
    n = state[3]
    w = state[4:7]
    
    dE = .5*(n*identity(3) + crux(E))@w
    dn = -.5*dot(E, w)

    Td = random.normal(0, 1e-3, (3,))

    dw = inv(I) @ (Td - crux(w)@ (I@w))
    
    wwdot = zeros(4)
    
    return hstack([dE, dn, dw, wwdot])
#propagateTruth

if __name__ == "__main__":
    main()
#if



