from numpy import *
from numpy.linalg import norm
from numpy.linalg import inv
from scipy import optimize
import datetime

def coes_to_RV(a, e, i, RAAN, omega, theta):
    # i, RAAN, omega, theta in degrees
    # a in km
    mu = 398600

    # Convert to radians:
    i = (pi/180)*i
    RAAN = (pi/180)*RAAN
    omega = (pi/180)*omega
    theta = (pi/180)*theta

    C = array([[cos(RAAN)*cos(omega) - sin(RAAN)*cos(i)*sin(omega), -cos(RAAN)*sin(omega) - sin(RAAN)*cos(i)*cos(omega), sin(RAAN)*sin(i)],
               [sin(RAAN)*cos(omega) + cos(RAAN)*cos(i)*sin(omega), -sin(RAAN)*sin(omega) + cos(RAAN)*cos(i)*cos(omega), -cos(RAAN)*sin(i)],
               [sin(i)*sin(omega), sin(i)*cos(omega), cos(i)]])

    R = C @ array([(a*(1-e**2)*cos(theta))/(1+e*cos(theta)), (a*(1-e**2)*sin(theta))/(1+e*cos(theta)), 0])
    V = C @ array([-sqrt(mu/(a*(1-e**2)))*sin(theta), sqrt(mu/(a*(1-e**2)))*(e + cos(theta)), 0])

    return R, V
# coes_to_RV

def RV_to_coes(R, V, mu):
    h = cross(R, V)
    r = norm(R)
    v = norm(V)
    v_r = dot(R, V) / r

    eps = (v ** 2 / 2) - (mu / r)
    a = -mu / (2 * eps)
    i = arccos(h[2] / norm(h))
    N = cross([0, 0, 1], h)
    n = norm(N)

    if N[1] >= 0:
        RAAN = arccos(N[0] / n)
    else:
        RAAN = 2*pi - arccos(N[0] / n)
    # end

    e_vec = (1 / mu) * array((cross(V, h) - (mu / r) * R))
    e = norm(e_vec)

    if e_vec[2] >= 0:
        omega = arccos(dot(N, e_vec) / (n * e))
    else:
        omega = 2*pi - arccos(dot(N, e_vec) / (n * e))
    # end


    if v_r >= 0:
        theta = arccos(dot(e_vec, R) / (e * r))
    else:
        theta = 2*pi - arccos(dot(e_vec, R) / (e * r))

    # Convert coes to degrees:
    i = (180/pi)*i
    RAAN = (180/pi)*RAAN
    omega = (180/pi)*omega
    theta = (180/pi)*theta

    return a, e, i, RAAN, omega, theta
# RV_to_coes


def Cz(ang):
    ang = radians(ang)
    c = cos(ang)
    s = sin(ang)
    C = array([[c,s,0],
               [-s,c,0],
               [0,0,1]])
    return C
# Cz

def Cy(ang):
    ang = radians(ang)
    c = cos(ang)
    s = sin(ang)
    C = array([[c,0,-s],
               [0,1,0],
               [s,0,c]])
    return C
# Cy

def Cx(ang):
    ang = radians(ang)
    c = cos(ang)
    s = sin(ang)
    C = array([[1,0,0],
               [0,c,s],
               [0,-s,c]])
    return C
# Cx

def find_sun(utc):
    JD = ut_to_jd(utc)
    n = JD - 2451545
    M = (357.529 + 0.98560023*n) % 360 # [deg]
    L = (280.459 + 0.98564736*n) % 360 # [deg]
    lam = (L + 1.915*sin((pi/180)*M) + 0.02*sin(2*(pi/180)*M)) % 360
    eps = 23.439 - (3.56e-7)*n
    u = array([cos((pi/180)*lam), sin((pi/180)*lam)*cos((pi/180)*eps), sin((pi/180)*lam)*sin((pi/180)*eps)])
    r = 149597870.691*(1.00014 - 0.01671*cos((pi/180)*M) - .00014*cos((pi/180)*2*M))
    return r*u # [km]
# find_sun

def ut_to_jd(utc):
    y = utc.year  # year
    m = utc.month  # month
    d = utc.day # day
    hr = utc.hour  # hour
    minute = utc.minute # min
    sec = utc.second  # sec
    j_0 = 367 * y - floor((7 * (y + floor((m + 9) / 12))) / 4) + floor((275 * m) / 9) + d + 1721013.5
    ut_hrs = hr + (minute / 60) + (sec / 3600)  # Universal time [hrs]
    return j_0 + ut_hrs/24
# ut_to_jd

def vec_cross(v):
    vx = v[0]
    vy = v[1]
    vz = v[2]
    r1 = array([0, -vz, vy])
    r2 = array([vz, 0, -vx])
    r3 = array([-vy, vx, 0])
    return vstack([r1, r2, r3])
# vec_cross

def angleBetween(a, b):
    a = a/norm(a)
    b = b/norm(b)
    return arccos(clip(dot(a, b), -1.0, 1.0))
# angleBetween

def UT2LST(utc, lon):
    # utc is a datetime object, lon in radians
    # Returns LST in radians
    # Extract data from utc object:
    y = utc.year  # year
    m = utc.month  # month
    d = utc.day # day
    hr = utc.hour  # hour
    minute = utc.minute # min
    sec = utc.second  # sec
    J0 = 367*y - floor((7*(y + floor((m + 9)/12)))/4) + floor((275*m)/9) + d + 1721013.5 # [days]
    date = datetime.datetime(2000, 1, 1, 11, 58, 56)
    J2000 = ut_to_jd(date) # [days]
    T0 = (J0 - J2000)/36525
    theta_G0 = (100.4606184 + 36000.77004*T0 + 0.000387933*T0**2 - (2.58*10**-8)*T0**3) % 360
    UT = hr + (minute/60) + (sec/3600) # [hrs]
    theta_G = theta_G0 + 360.98564724*(UT/24) # [deg]
    LST = (theta_G + lon*180/pi) % 360 # [deg]
    return LST*pi/180
# UT2LST

def findSite(lat, lst, alt):
    # angles in radians
    f = .003353
    a = ((6378/sqrt(1 - (2*f - f**2)*(sin(lat)**2))) + alt)*cos(lat)
    b = (6378*(1 - f)**2)/(sqrt(1 - (2*f - f**2)*(sin(lat)**2))) + alt
    x = a*cos(lst)
    y = a*sin(lst)
    z = b*sin(lat)
    return array([x, y, z])
# findSite

def AzEl2RaDec(Az, El, lat, lon, utc):
    # Az, El, lat, lon in radians
    # Outputs Ra, dec in radians
    dec = arcsin(sin(El)*sin(lat) + cos(El)*cos(lat)*cos(Az))
    LST = UT2LST(utc, lon)
    LHA = arctan2(-sin(Az)*cos(El)/cos(dec), (sin(El) - sin(dec)*sin(lat))/(cos(dec)*cos(lat)))
    Ra = ((180/pi)*(LST - LHA)) % 360
    return Ra*pi/180, dec
# AzEl2RaDec

def RaDec2AzEl(Ra, Dec, lat, lon, utc):
    # Ra, Dec, lat, lon in radians
    # Outputs Ra, dec in radians
    LST = (180/pi)*UT2LST(utc, lon)
    LHA = (pi/180)*((LST - Ra*180/pi) % 360)
    El = arcsin(sin(lat)*sin(Dec) + cos(lat)*cos(Dec)*cos(LHA))
    Az = arctan2(-sin(LHA)*cos(Dec)/cos(El), (sin(Dec) - sin(El)*sin(lat))/(cos(El)*cos(lat)))
    return Az, El
# RaDec2AzEl

def gaussDetermination(sites, times, observations, ext):
    mu = 398600
    # ALL angles in radians

    # Extract data:
    RS_1 = sites[0]
    RS_2 = sites[1]
    RS_3 = sites[2]

    Ra_1 = observations[0][0]
    Dec_1 = observations[0][1]

    Ra_2 = observations[1][0]
    Dec_2 = observations[1][1]

    Ra_3 = observations[2][0]
    Dec_3 = observations[2][1]

    # Get unit vectors from site to object in ECI:
    rho_1 = array([cos(Dec_1)*cos(Ra_1), cos(Dec_1)*sin(Ra_1), sin(Dec_1)])
    rho_2 = array([cos(Dec_2)*cos(Ra_2), cos(Dec_2)*sin(Ra_2), sin(Dec_2)])
    rho_3 = array([cos(Dec_3)*cos(Ra_3), cos(Dec_3)*sin(Ra_3), sin(Dec_3)])

    # Build the matrices:
    L = array([rho_1, rho_2, rho_3]).transpose()
    R = array([RS_1, RS_2, RS_3]).transpose()

    Tau_1 = (times[0] - times[1]).total_seconds()
    Tau_3 = (times[2] - times[1]).total_seconds()

    a1 = Tau_3/(Tau_3 - Tau_1)
    a3 = -Tau_1/(Tau_3 - Tau_1)
    a1u = Tau_3*((Tau_3 - Tau_1)**2 - Tau_3**2)/(6*(Tau_3 - Tau_1))
    a3u = -Tau_1*((Tau_3 - Tau_1)**2 - Tau_1**2)/(6*(Tau_3 - Tau_1))

    M = inv(L) @ R

    d1 = M[1][0]*a1 - M[1][1] + M[1][2]*a3
    d2 = M[1][0]*a1u + M[1][2]*a3u

    C = dot(L.transpose()[1], RS_2)

    func = eighthOrderGarbage(d1, d2, C, mu, norm(RS_2))
    root = optimize.fsolve(func, 1e6)[0]

    r2 = root
    u = mu/(r2**3)

    c1 = a1 + a1u*u
    c2 = -1
    c3 = a3 + a3u*u

    A = M @ array([-c1, -c2, -c3])

    Rho_1 = A[0]/c1
    Rho_2 = A[1]/c2
    Rho_3 = A[2]/c3

    # Obtain position vectors:
    R1 = Rho_1*rho_1 + RS_1
    R2 = Rho_2*rho_2 + RS_2
    R3 = Rho_3*rho_3 + RS_3

    # Calculate lagrange coefficients:
    f1 = 1 - (u/2)*Tau_1**2
    g1 = Tau_1 - (u/6)*Tau_1**3
    f3 = 1 - (u/2)*Tau_3**2
    g3 = Tau_3 - (u/6)*Tau_3**3

    # Get velocity vector:
    V2 = (1/(f1*g3 - g1*f3))*(f1*R3 - f3*R1)

    if ext == 1:
        err = 1
        n = 0
        while err > 0.001 or n > 1000:
            r2 = norm(R2)
            v2 = norm(V2)
            alpha = (2/r2) - (v2**2)/mu # Reciprocal of semi-major axis [1/km]
            v2r = dot(R2, V2)/r2
            func1 = kepler(r2, v2r, Tau_1, alpha)
            func3 = kepler(r2, v2r, Tau_3, alpha)

            chi0_1 = sqrt(mu)*absolute(alpha)*Tau_1
            chi0_3 = sqrt(mu)*absolute(alpha)*Tau_3

            chi_1 = optimize.fsolve(func1, chi0_1)[0]
            chi_3 = optimize.fsolve(func3, chi0_3)[0]

            z_1 = alpha*chi_1**2
            z_3 = alpha*chi_3**2

            f1 = (1/2)*(f1 + 1 - ((chi_1**2)/r2)*stumpffC(z_1))
            g1 = (1/2)*(g1 + Tau_1 - (1/sqrt(mu))*(chi_1**3)*stumpffS(z_1))
            f3 = (1/2)*(f3 + 1 - ((chi_3**2)/r2)*stumpffC(z_3))
            g3 = (1/2)*(g3 + Tau_3 - (1/sqrt(mu))*(chi_3**3)*stumpffS(z_3))

            c1 = g3/(f1*g3 - f3*g1)
            c3 = -g1/(f1*g3 - f3*g1)

            A = M @ array([-c1, -c2, -c3])

            # Calculate new range estimates:
            Rho_1_new = A[0]/c1
            Rho_2_new = A[1]/c2
            Rho_3_new = A[2]/c3

            # Calculate new position estimates:
            R1 = Rho_1_new*rho_1 + RS_1
            R2 = Rho_2_new*rho_2 + RS_2
            R3 = Rho_3_new*rho_3 + RS_3

            V2 = (1/(f1*g3 - g1*f3))*(f1*R3 - f3*R1)

            err = sqrt(absolute(Rho_1_new - Rho_1)**2 + absolute(Rho_2_new - Rho_2)**2 + absolute(Rho_2_new - Rho_2)**2)

            Rho_1 = Rho_1_new
            Rho_2 = Rho_2_new
            Rho_3 = Rho_3_new
            n = n + 1
        #while

        R2 = Rho_2*rho_2 + RS_2
        V2 = (1/(f1*g3 - g1*f3))*(f1*R3 - f3*R1)
        a, e, i, RAAN, omega, theta = RV_to_coes(R2, V2)

    else:
        a, e, i, RAAN, omega, theta = RV_to_coes(R2, V2)
    # else
    return a, e, i, RAAN, omega, theta
# guassDetermination


def eighthOrderGarbage(d1, d2, C, mu, r2_site):
    def func(r2):
        return r2**8 - (d1**2 + 2*C*d1 + r2_site**2)*r2**6 - 2*mu*(C*d2 + d1*d2)*r2**3 - (mu**2)*d2**2
    #func
    return func
#eighthOrderGarbage


def stumpffC(zeta):
    if zeta == 0:
        return 1/2
    elif zeta > 0:
        return (1 - cos(sqrt(zeta)))/zeta
    else:
        return -(cosh(sqrt(-zeta)) - 1)/zeta
    #else
#stumpffC

def stumpffS(zeta):
    if zeta == 0:
        return 1/6
    elif zeta > 0:
        return (sqrt(zeta) - sin(sqrt(zeta)))*(zeta)**(-3/2)
    else:
        return (sinh(sqrt(-zeta)) - sqrt(-zeta))*(-zeta)**(-3/2);
    #else
#stumpffS

def kepler(R, V_0r, dt, alpha):
    u = 398600
    def func(chi):
        z = alpha*chi**2
        return ((R*V_0r)/sqrt(u))*stumpffC(z)*chi**2 + (1 - alpha*R)*stumpffS(z)*chi**3 + R*chi - sqrt(u)*dt
    #func
    return func
#kepler

def fZeta(r1, r2, A, dt):
    mu = 398600
    def func(z):
        return stumpffS(z)*(((r1 + r2 + A*(z*stumpffS(z) - 1)/sqrt(stumpffC(z))))/stumpffC(z))**(3/2) + A*sqrt(((r1 + r2 + A*(z*stumpffS(z) - 1)/sqrt(stumpffC(z))))) - sqrt(mu)*dt
    #func
    return func
#fZeta

def lambertUV(R1, R2, dt, mu):
    r1 = norm(R1)
    r2 = norm(R2)
    if cross(R1, R2)[2] >= 0:
        dtheta = angleBetween(R1, R2)
    else:
        dtheta = 2*pi - angleBetween(R1, R2)
    #else
    A = sin(dtheta)*sqrt(r1*r2/(1 - cos(dtheta)))
    z_upper = 4*pi**2
    z_lower = -4*pi
    c = 1/2
    s = 1/6
    z = 0
    Y = r1 + r2 + A*(z*s - 1)/sqrt(c)
    chi = sqrt(Y/stumpffC(z))
    dt_n = (chi**3*stumpffS(z) + A*sqrt(Y))/sqrt(mu)
    while absolute(dt_n - dt) >= 1e-5:
        if dt_n <= dt:
            z_lower = z
        else:
            z_upper = z
        #else
        z = (1/2)*(z_upper + z_lower)
        c = stumpffC(z)
        s = stumpffS(z)
        Y = r1 + r2 + A*(z*s - 1)/sqrt(c)
        chi = sqrt(Y/stumpffC(z))
        dt_n = (chi**3*stumpffS(z) + A*sqrt(Y))/sqrt(mu)
    #while
    f = 1 - Y/r1
    gdot = 1 - Y/r2
    g = A*sqrt(Y/mu)
    V2 = (1/g)*(gdot*R2 - R1)
    V1 = (1/g)*(R2 - f*R1)
    return V1, V2
#lambertUV

def lambertGauss(R1, R2, dt, mu):
    # Assumes short way around
    r1 = norm(R1)
    r2 = norm(R2)
    theta = arccos(dot(R1, R2)/(r1*r2)) # [rad]
    l = ((r1 + r2)/(4*sqrt(r1*r2)*cos(theta/2))) - (1/2)
    m = mu*(dt**2)/((2*sqrt(r1*r2)*cos(theta/2))**3)
    y = 1
    err = 1
    while err >= 1e-3:
        x1 = (m/y**2) - l
        x2 = (4/3)*(1 + (6/5)*x1  + (48/35)*x1**2 + (480/315)*x1**3)
        y_new = 1 + x2*(l + x1)
        err = absolute(y - y_new)
        y = y_new
    #while
    cos_E_half = 1 - 2*x1
    p = (r1*r2)*(1 - cos(theta))/(r1 + r2 - 2*sqrt(r1*r2)*cos(theta/2)*cos_E_half)
    f = 1 - (r2/p)*(1 - cos(theta))
    g = r1*r2*sin(theta)/sqrt(mu*p)
    V1 = (R2 - f*R1)/g
    gdot = 1 - (r1/p)*(1 - cos(theta))
    V2 = (1/g)*(gdot*R2 - R1)
    return V1, V2
#lambertGauss

def lambertMinE(R1, R2, dt):
    # Assumes 1 rev, short way around
    mu = 398600
    r1 = norm(R1)
    r2 = norm(R2)
    cos_theta = dot(R1, R2)/(r1*r2)
    sin_theta = sqrt(1 - cos_theta**2)
    c = sqrt(r1**2 + r2**2 - 2*r1*r2*cos_theta)
    s = (1/2)*(r1 + r2 + c)
    a_min = s/2
    beta_min = 2*arcsin(sqrt(1 - c/s))
    t_min = sqrt(a_min**3/mu)*(pi - (beta_min - sin(beta_min)))
    p_min = (r1*r2/c)*(1 - cos_theta)
    V1 = (sqrt(mu*p_min)/(r1*r2*sin_theta))*(R2 - (1 - (r1/p_min)*(1 - cos_theta))*R1)
    return V1
#lambertMinE

def ME2TA(Me, ecc):
    Me = radians(Me)

    f = lambda x: x - ecc*sin(x) - Me

    if Me < pi:
        guess = Me + ecc
    else:
        guess = Me - ecc

    E = optimize.fsolve(f,guess,xtol = 1e-9)[0]

    angle = 2*arctan(tan(E/2)/sqrt((1-ecc)/(1+ecc)))
    angle = degrees(angle)
    return angle
#ME2TA

def tle2coes(filename):
    file = open(filename)
    coes = {}
    mu = 398600
    for line in file.readlines():
        elements = line.split()
        if elements[0] == '1':
            year = 2000 + int(elements[3][:2])
            daynumber = float(elements[3][2:])
            time = datetime.datetime(year = year, month = 1, day = 1)
            time += datetime.timedelta(days = daynumber - 1)
            coes['time'] = time
        elif elements[0] == '2':
            #In degrees
            inc = float(elements[2])
            RAAN = float(elements[3])
            arg_p = float(elements[5])
            mean_anomaly = float(elements[6])
            ecc = float('.'+elements[4])
            mean_motion = float(elements[7])

            TA = ME2TA(mean_anomaly,ecc)%360
            Period = 24*3600/mean_motion
            mean_motion = (2*pi/86400)*mean_motion
            semi_a = (mu/mean_motion**2)**(1/3)

            coes['i'] = inc
            coes['RAAN'] = RAAN
            coes['omega'] = arg_p
            coes['theta'] = TA
            coes['e'] = ecc
            coes['T'] = Period
            coes['a'] = semi_a
    return coes
#tle2coes

def atm_dens(z):
    #Height ranges
    h = [ 0, 25, 30, 40, 50, 60, 70,
         80, 90, 100, 110, 120, 130, 140,
         150, 180, 200, 250, 300, 350, 400,
         450, 500, 600, 700, 800, 900, 1000]
    #Densities
    r = [1.225, 4.008e-2, 1.841e-2, 3.996e-3, 1.027e-3, 3.097e-4, 8.283e-5,
         1.846e-5, 3.416e-6, 5.606e-7, 9.708e-8, 2.222e-8, 8.152e-9, 3.831e-9,
         2.076e-9, 5.194e-10, 2.541e-10, 6.073e-11, 1.916e-11, 7.014e-12, 2.803e-12,
         1.184e-12, 5.215e-13, 1.137e-13, 3.070e-14, 1.136e-14, 5.759e-15, 3.561e-15]
    #Scale height
    H = [ 7.310, 6.427, 6.546, 7.360, 8.342, 7.583, 6.661,
         5.927, 5.533, 5.703, 6.782, 9.973, 13.243, 16.322,
         21.652, 27.974, 34.934, 43.342, 49.755, 54.513, 58.019,
         60.980, 65.654, 76.377, 100.587, 147.203, 208.020]

    if z > 1000:
        z = 1000
    elif z < 0:
        z = 0

    i = 0
    #Determine the interpolation interval:
    for j in range(0,26):
        if z >= h[j] and z < h[j+1]:
            i = j;

    if z >= 1000:
        i = 26

    density = r[i]*exp(-(z - h[i])/H[i])

    return density
#atmosDensity

def R2RaDec(rho): # This needs to be done on rho
    rho_hat = rho/norm(rho)
    l = rho_hat[0]
    m = rho_hat[1]
    n = rho_hat[2]
    dec = arcsin(n)
    alpha = arccos(l/cos(dec))
    if  m <= 0:
        alpha = 2*pi - alpha
    #if
    return alpha, dec
#R2RaDec

def isInSun(R, R_sun):
    theta = arccos(dot(R, R_sun)/(norm(R)*norm(R_sun)))
    theta1 = arccos(6378/norm(R))
    theta2 = arccos(6378/norm(R_sun))
    if (theta1 + theta2) <= theta:
        return 0 # eclipsed
    else:
        return 1 # in sunlight
#isInSun

def twoBody(t, state, mu):
    R = state[0:3]
    V = state[3:6]
    a = -(mu/norm(R)**3)*R
    return hstack([V, a])
#twoBody

def QtoC(q):
    eps = q[0:3]
    eta = q[3]
    return (2*eta**2 - 1)*identity(3) + 2*outer(eps, eps) - 2*eta*vecCross(eps)
# QtoC

def CtoQ( C ):
    E = zeros(3)
    n = 0
    tr = trace(C)

    if (tr > 0):
        n = sqrt( tr + 1 )/2

        E[0] = (C[1, 2] - C[2, 1])/(4*n)
        E[1] = (C[2, 0] - C[0, 2])/(4*n) 
        E[2] = (C[0, 1] - C[1, 0])/(4*n) 
    else:
        d = diag(C)
        if max(d) == d[1]:

            sq_trace = sqrt(d[1] - d[0] - d[2] + 1 )

            E[1] = .5*sq_trace 

            if sq_trace != 0:
                sq_trace = .5/sq_trace

            n    = (C[2, 0] - C[0, 2])*sq_trace 
            E[0] = (C[0, 1] + C[1, 0])*sq_trace
            E[2] = (C[1, 2] + C[2, 1])*sq_trace

        elif max(d) == d[2]:
            sq_trace = sqrt(d[2] - d[0] - d[1] + 1)

            E[2] = .5*sq_trace 

            if sq_trace != 0:
                sq_trace = .5/sq_trace

            n    = (C[0, 1] - C[1, 0])*sq_trace
            E[0] = (C[2, 0] + C[0, 2])*sq_trace 
            E[1] = (C[1, 2] + C[2, 1])*sq_trace
        else:
            sq_trace = sqrt(d[0] - d[1] - d[2] + 1)

            E[0] = .5*sq_trace 

            if sq_trace != 0:
                sq_trace = .5/sq_trace

            n    = (C[1, 2] - C[2, 1])*sq_trace 
            E[1] = (C[0, 1] + C[1, 0])*sq_trace
            E[2] = (C[2, 0] + C[0, 2])*sq_trace

    return hstack([E, n])


def vecCross(v):
    r1 = array([0, -v[2], v[1]])
    r2 = array([v[2], 0, -v[0]])
    r3 = array([-v[1], v[0], 0])
    return vstack([r1, r2, r3])
#vecCross

def crux(A):
    return array([[0, -A[2], A[1]],
                  [A[2], 0, -A[0]],
                  [-A[1],A[0], 0]])
# crux

def propagateEKF(t, state, I):
    E = state[0:3]
    n = state[3]
    w = state[4:7]
    dE = .5*(n*identity(3) + crux(E)) @ w
    dn = -.5*dot(E, w)
    dw = inv(I) @ (-crux(w) @ (I@w))
    return hstack([dE, dn, dw])
# propagateEKF

def quat_mult(E1, n1, E2, n2):

    # q1 *q2

    E3 = n1*E2 + n2*E1 + crux(E1)@E2
    n3 = n2*n1 - dot(E1, E2)

    return E3, n3
#quat_mult

def makeAttitudeFile(time, quats, filename):
    file = open(filename, 'w')
    file.write('stk.v.11.0\n'
                'BEGIN Attitude\n'
                'NumberOfAttitudePoints  '+str(len(time))+'\n'
                'ScenarioEpoch           22 May 2019 19:00:00.000000000\n'
                'BlockingFactor          20\n'
                'InterpolationOrder      1\n'
                'CentralBody             Moon\n'
                'CoordinateAxes          J2000\n'
                'AttitudeTimeQuaternions\n')
    for t, quat in zip(time, quats):
        file.write('{:.12E} {:.12E} {:.12E} {:.12E} {:.12E}\n'.format(t, quat[0], quat[1], quat[2], quat[3]))
    #for
    file.write('END Attitude')
#makeAttitudeFile
