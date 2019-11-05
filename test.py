from StarTrackerFilter import *
from AERO_TOOLKIT_NEW import *
from pyquaternion import *
import matplotlib.pyplot as plt
from jplephem.spk import SPK
from Controller import *
from datetime import datetime
from datetime import timedelta

def main():

    # Initial state:
    n = 0.5
    E = array([0.5, 0.5, 0.5])
    w = array([0.001, -0.002, 0.001])
    w_wheels = zeros(4)
    mu = 398600 # [km^3/s^2]
    R_mag = 384400 + 61350
    R = array([R_mag, 0, 0]) # [km]
    V = array([0, sqrt(mu/norm(R)), 0])
    T = 2*pi*sqrt(norm(R)**3/mu)

    # SC Properties:
    I = diag(array([1490, 940, 1520])) # Inertia matrix in principal body frame [kg*m^2]

    # Body frame to principle frame inertia matrix:
    C_PRINC_BODY = identity(3)

    utc = datetime(year = 2019, month = 5, day = 22, hour = 19, minute = 0, second = 0)

    # Wheel properties:
    m = 15 # [kg]
    r = .1 # [m]
    h = .01 # [m]
    Iw = (1/2)*m*r**2 # Inertia of wheels around spin axis [kg*m^2]

    dt = .5
    tspan = 15*60 + 60*60
    tspan = int(tspan) # Total simulation time [sec]

    # Noise estimates (standard deviations) for EKF:
    STAR_TRACKER_NOISE = 9e-4 # [rad]
    GYRO_NOISE = 1e-3 # [rad/s]
    MEASUREMENT_NOISE = diag(hstack([[STAR_TRACKER_NOISE]*4, [GYRO_NOISE]*3]))
    PROCESS_NOISE = sqrt(1e-7**2)*identity(7)
    COVARIANCE_GUESS = sqrt(1e-7**2)*identity(7)
    
    POS_NOISE = sqrt(1e-2) # [km^1/2]
    VEL_NOISE = sqrt(1e-2) # [(km/s)^1/2]

    # Initial EKF estimate:
    aguess = array([2, 4, 3])/norm(array([2, 4, 3]))
    thetaguess = 1.6
    Eguess = aguess*sin(thetaguess/2)
    nguess = cos(thetaguess/2)
    w_guess = array([0.07, -0.1, 0.05])

    # Instantiate EKF object:
    EKF = StarTracker_Filter(Eguess, nguess, w_guess, I, Iw, PROCESS_NOISE, MEASUREMENT_NOISE, COVARIANCE_GUESS)

    # Actuator Properties and Configuration:
    WHEEL_TILT = 45*(pi/180) # [rad]
    WHEEL_INERTIAS = diag([Iw]*4) # [kg*m^2]
    DAMPING_RATIO = .7
    SETTLING_TIME = 5*60 # [sec]
    s = sin(WHEEL_TILT)
    c = cos(WHEEL_TILT)
    AS = array([[s, 0,-s, 0],
                [0, s, 0,-s],
                [c, c, c, c]]) 

    # Instantiate Controller Object:
    R0_guess = R + random.normal(0, POS_NOISE, (3,))
    V0_guess = V + random.normal(0, VEL_NOISE, (3,))

    controller = Controller(I, WHEEL_INERTIAS, AS, C_PRINC_BODY)
    KP, KD, = controller.calc_gains(DAMPING_RATIO, SETTLING_TIME)
    controller.set_gains(KP, KD)
    controller.set_mode("Nadir")

    state = hstack([E, n, w, w_wheels, R, V])

    solver = ode(propagateTruth)
    solver.set_integrator('lsoda', max_step = dt, atol = 1e-9, rtol = 1e-9)
    solver.set_initial_value(state, 0)
    solver.set_f_params(I, WHEEL_INERTIAS, AS, mu, zeros(3), zeros(4))

    num_pts = int(tspan/dt)

    t = zeros(num_pts) # [sec]
    newstate = zeros((num_pts, 17))
    measurements = zeros((num_pts, 7))
    state_estimate = zeros((num_pts, 7))
    ang_vel_error = zeros((num_pts, 3))
    q_error = zeros((num_pts, 4))
    pointing_error = zeros(num_pts)
    utcs = []
    angle_off_target = zeros(num_pts)
    gradient_torques = zeros((num_pts, 3))

    Tc = zeros(3)
    energy = 0

    kernel = SPK.open('de430.bsp') 
    EARTH = 3
    MOON = 301
    SUN = 10
    CENTER = 0
    a = array([0, 0, 1]) # Boresight vector in body frame

    percentage = 10

    tumble_time = 2*60

    # Solve:
    for i in range(num_pts):

        # Simulate measurements:
        q_true = Quaternion(array = array([solver.y[3], solver.y[0], solver.y[1], solver.y[2]]))
        dq = Quaternion(array = hstack([1, random.normal(0, STAR_TRACKER_NOISE, (3,))/2]))
        q_measurement = dq * q_true
        q_measurement = array([q_measurement[1], q_measurement[2], q_measurement[3], q_measurement[0]])
        w_measurement = solver.y[4:7] + random.normal(0, GYRO_NOISE, (3,))

        # Update the filter:
        measurement = hstack([q_measurement, w_measurement])
        
        w_wheels = solver.y[7:11]
        R = solver.y[11:14] + random.normal(0, POS_NOISE, (3,))
        V = solver.y[14:] + random.normal(0, VEL_NOISE, (3,))

        C_princ_inertial_true = QtoC(solver.y[0:4])
        estimate = EKF.update(measurement, dt, C_princ_inertial_true@R, Tc)

        eps = estimate[0:3]
        eta = estimate[3]
        w = estimate[4:7]

        if solver.t > tumble_time:
            Tc = controller.command_torque(eps, eta, w, R, V, utc, a)
            wheel_accel = controller.command_wheel_acceleration(Tc)
            solver.set_f_params(I, WHEEL_INERTIAS, AS, mu, Tc, wheel_accel) 
        #if

        # Pointing error calc:
        C_princ_inertial_estimate = QtoC(estimate[0:4])
        z_I_estimate = C_princ_inertial_estimate[2, :]
        z_I_true = C_princ_inertial_true[2, :]

        #gravity gradient torque
        Rprinc = C_princ_inertial_true @ R
        Tgg = (3*mu/norm(R)**5)*(crux(Rprinc) @ (I @ Rprinc))

        # Kalman Quaternion error calc:
        q_estimate = Quaternion(array = array([estimate[3], estimate[0], estimate[1], estimate[2]]))
        q_e = q_true.conjugate*q_estimate

        # Integrate:
        solver.integrate(solver.t + dt)

        # increment time
        utc += timedelta(seconds = dt)

        # Ephemerides:
        R_sun = kernel[CENTER, SUN].compute(ut_to_jd(utc)) # Center of Solar System to Sun
        R_earth = kernel[CENTER, EARTH].compute(ut_to_jd(utc)) # Center of Solar System to Earth

        # Earth calcs:
        R_earth_moon = kernel[EARTH, MOON].compute(ut_to_jd(utc)) # Earth to Moon
        R_sc_earth = -R - R_earth_moon # Spacecraft to Earth

        # Sun calcs:
        R_sc_sun = R_sun - R_earth - R_earth_moon - R

        if (solver.t < tspan/3):
            controller.set_mode("Nadir")
            target =-R
        elif (solver.t >= tspan/3) and (solver.t < tspan*2/3):
            controller.set_mode("Earth_Point")
            target = -R_sc_earth
        elif (solver.t >= tspan*2/3):
            controller.set_mode("Sun_Point")
            target = -R_sc_sun
        # #elif

        #target = -solver.y[11:14]

        #save data:
        t[i] = solver.t
        newstate[i] = solver.y
        q_error[i] = array([q_e[1], q_e[2], q_e[3], q_e[0]])
        pointing_error[i] = angleBetween(z_I_true, z_I_estimate)
        ang_vel_error[i] = norm(estimate[4:7]) - norm(solver.y[4:7])
        state_estimate[i] = estimate
        measurements[i] = measurement
        utcs.append(utc)
        gradient_torques[i] = Tgg
        angle_off_target[i] = angleBetween(C_PRINC_BODY @ a, C_princ_inertial_true @ target)

        # Progress:
        completion = solver.t/tspan*100
        if completion > percentage:
            print(percentage,"percent complete")
            percentage+= 10
        #if
        
    #for
    print(percentage,"percent complete")

    makeAttitudeFile(t, newstate[:, 0:4], 'test.a')

    plt.close()

    print("Plotting...")

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
    plt.savefig('Quaternion Estimate vs Truth.png', bbox_inches = 'tight', dpi = 1000)

    print("1/8")

    fig1 = plt.figure()

    plt.plot(t/T, q_error[:, 0])
    plt.plot(t/T, q_error[:, 1])
    plt.plot(t/T, q_error[:, 2])
    plt.plot(t/T, q_error[:, 3])
    plt.grid()
    plt.title('Kalman Filter Quaternion Estimate Error')
    plt.xlabel('Time [Number of Orbits]')
    plt.legend(['q_i', 'q_j', 'q_k', 'q_r'])
    plt.savefig('Kalman Filter Quaternion Estimate Error.png', bbox_inches = 'tight', dpi = 1000)

    print("2/8")

    fig2 = plt.figure()

    w1_error = newstate[:, 4] - state_estimate[:,4]
    w2_error = newstate[:, 5] - state_estimate[:,5]
    w3_error = newstate[:, 6] - state_estimate[:,6]

    plt.plot(t/T, (180/pi)*w1_error)
    plt.plot(t/T, (180/pi)*w2_error)
    plt.plot(t/T, (180/pi)*w3_error)
    plt.grid()
    plt.title('Kalman Angular Velocity Error [deg/s]')
    plt.xlabel('Time [Number of Orbits]')
    plt.legend(['Body X', 'Body Y', 'Body Z'])
    plt.savefig('Kalman Angular Velocity Error.png', bbox_inches = 'tight', dpi = 1000)

    print("3/8")

    fig3 = plt.figure()

    plt.semilogy(t/T, (180/pi)*pointing_error, '.')
    plt.grid()
    plt.title('Kalman Pointing Knowledge Error [degrees]')
    plt.xlabel('Time [Number of Orbits]')
    plt.savefig('Kalman Pointing Knowledge Error.png', bbox_inches = 'tight', dpi = 1000)

    print("4/8")

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
    plt.savefig('Body Rates.png', bbox_inches = 'tight', dpi = 1000)

    print("5/8")

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
    plt.savefig('Wheel Momentum Storage.png', bbox_inches = 'tight', dpi = 1000)

    print("6/8")

    fig6, axes6 = plt.subplots(2, 1, squeeze = False)
    axes6[1][0].semilogy(t/T, angle_off_target*180/pi)
    axes6[1][0].semilogy([t[0]/T, tspan/T/4, tspan/T/4, tspan/T/2, tspan/T/2, t[-1]/T], [1, 1 ,1, 1, 1, 1],'r')
    axes6[1][0].grid()
    axes6[1][0].set_xlabel('Time [Number of Orbits]')
    axes6[1][0].set_ylabel('Angle Error [deg]')

    axes6[0][0].plot([0, tumble_time/T, tumble_time/T, tspan/T/3, tspan/T/3, tspan/T*2/3, tspan/T*2/3, tspan/T], [0 ,0 ,1 ,1 ,2 ,2 ,3 , 3])
    axes6[0][0].set_yticks([0,1,2,3])
    axes6[0][0].set_yticklabels(['Tumble', 'Earth Point', 'Nadir Point', 'Sun Point'])
    axes6[0][0].set_title('Mode')
    fig6.suptitle('Angle from Target')

    plt.savefig('Angle from Target.png', bbox_inches = 'tight', dpi = 1000)

    print("7/8")

    fig7 = plt.figure()
    plt.plot(t/T, gradient_torques)
    plt.grid()
    plt.title('Gravity Gradient Torques')
    plt.xlabel('Time [Number of Orbits]')
    plt.ylabel('Torque [Nm]')
    plt.savefig('Gravity Gradient Torques.png', bbox_inches = 'tight', dpi = 1000)

    fig8 = plt.figure()

    print("8/8")
    
    plt.plot(t / T, (60/(2*pi))*newstate[:, 7])
    plt.plot(t / T, (60/(2*pi))*newstate[:, 8])
    plt.plot(t / T, (60/(2*pi))*newstate[:, 9])
    plt.plot(t / T, (60/(2*pi))*newstate[:, 10])
    plt.grid()
    plt.title('Wheel Angular Velocity')
    plt.xlabel('Time [Number of Orbits]')
    plt.ylabel('Angular Velocity [RPM]')
    plt.legend(['1', '2', '3', '4'])
    plt.savefig('Wheel Angular Velocity.png', bbox_inches = 'tight', dpi = 1000)

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

    C_princ_inertial = QtoC(hstack([E, n]))
    Rprinc = C_princ_inertial @ R
    Tgg = (3*mu/norm(R)**5)*(crux(Rprinc) @ (I @ Rprinc)) 
    # Td_random = random.normal(0, 1e-8, (3,)) 
    h_w = As @ (Ics @ w_wheels)
    dw = inv(I) @ (Tc + Tgg - crux(w) @ (I@w + h_w))

    wwdot = wheel_accel

    return hstack([dE, dn, dw, wwdot, dR, dV])
#propagateTruth

if __name__ == "__main__":
    main()
#if