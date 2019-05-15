from numpy import *
from numpy.linalg import *

# Ixx = 11232.05
# Iyx = -80.69
# Izx = -21.24
# Iyy = 10177.19
# Izy = -177.78
# Izz = 2634.27

# Ibody = array([[Ixx, Iyx, Izx],
# 			   [Iyx, Iyy, Izy],
# 			   [Izx, Izy, Izz]])

# k, V = eig(Ibody)

# print(V)
# print(k)

# C = identity(3)

# q = Rotation.from_C(C).as_quat()
# print(q)


def dcm2quat( C ):
	n = 0
	tr = trace(C)

	if (tr > 0):
		n = sqrt( tr + 1 )/2

		E[0] = (C[2, 3] - C[3, 2])/(4*n)
		E[1] = (C[3, 1] - C[1, 3])/(4*n) 
		E[2] = (C[1, 2] - C[2, 1])/(4*n) 
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

			if sqdip1 != 0:
				sq_trace = .5/sq_trace

			n    = (C[1, 2] - C[2, 1])*sq_trace 
			E[1] = (C[0, 1] + C[1, 0])*sq_trace
			E[2] = (C[2, 0] + C[0, 2])*sq_trace

	return E, n
