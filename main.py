from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from kalman_filter import KalmanFilter

def generateTestData():
	n = 1000
	theta = linspace(-4 * pi, 4 * pi, n)
	z = linspace(-2, 2, n)
	r = z**2 + 1
	z = z + randn(n) * 0.1
	x = r * sin(theta)
	y = r * cos(theta)
	return x, y, z

def test(x, y, z, qcov, rcov):
	# Initializations
	dt = 1 									# Measurement interval (s)
	I3 = identity(3)
	A = zeros((9,9)) 						# Transition matrix
	A[0:3, 0:3] = I3
	A[0:3, 3:6] = I3 * dt
	A[0:3, 6:9] = I3 * 0.5 * dt * dt
	A[3:6, 3:6] = I3
	A[3:6, 6:9] = I3 * dt
	A[6:9, 6:9] = I3
	H = zeros((3,9)) 						# Measurement matrix
	H[0:3, 0:3] = I3
	Q = identity(9) * qcov					# Transition covariance
	R = identity(3) * rcov					# Noise covariance
	B = identity(9)							# Control matrix
	kf = KalmanFilter(A, H, Q, R, B)
	# Run through the dataset
	n = len(x)
	xkf, ykf, zkf = zeros(n), zeros(n), zeros(n)
	for i in xrange(n):
		kf.correct(array([x[i], y[i], z[i]]))
		kf.predict(array([]))
		Skf = kf.getCurrentState()
		xkf[i], ykf[i], zkf[i] = Skf[0], Skf[1], Skf[2]
	return xkf, ykf, zkf

def main():
	# Compare over Q
	qcov = [1e-2, 1e-6]
	rcov = [1, 1]
	# Compare over R
	qcov = [1e-2, 1e-2]
	rcov = [1, 10000]
	# Best
	qcov = [1e-4]
	rcov = [1000]
	x, y, z = generateTestData()
	fig = figure()
	ax = fig.gca(projection='3d')
	ax.plot(x, y, z, label='Measurements')
	for r, q in zip(rcov, qcov):
		xkf, ykf, zkf = test(x, y, z, q, r)
		print r, q
		ax.plot(xkf, ykf, zkf, label='Rcov = %f, Qcov = %f' % (r, q))
	ax.legend()
	show()

if __name__ == '__main__':
 	main() 