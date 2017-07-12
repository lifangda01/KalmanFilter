from pylab import *

class KalmanFilter(object):
	"""Basic linear kalman filter"""
	def __init__(self, A, H, Q, R, B):
		super(KalmanFilter, self).__init__()
		self.A = A 				# Transition matrix
		self.H = H 				# Measurement matrix
		self.Q = Q 				# Transition covariance
		self.R = R				# Noise covariance
		self.B = B 				# Control matrix
		self.P = array([]) 		# State covariance
		self.x = array([]) 		# State vector
		self.xPrior = array([])
		self.PPrior = array([])

	def predict(self, u):
		"""
			Prediction step given control matrix B.
		"""
		# Project the state ahead
		if len(u) == 0:
			self.xPrior = dot(self.A, self.x)
		else:
			self.xPrior = dot(self.A, self.x) + dot(self.B, u)
		# Project the covariance
		self.PPrior = dot(self.A, dot(self.P, self.A.transpose())) + self.Q

	def correct(self, z):
		"""
			Correction step given measurement z.
		"""
		# If this is the first measurement...
		if len(self.x) == 0:
			self.x = hstack((z, zeros(6)))
			self.P = identity(size(self.x))
			self.xPrior = copy(self.x)
			self.PPrior = copy(self.P)
		# Compute the Kalman gain
		M1 = dot(self.PPrior, self.H.transpose())
		M2 = dot(self.H, dot(self.PPrior, self.H.transpose())) + self.R
		K = dot(M1, inv(M2))
		# Update state estimate
		v1 = z - dot(self.H, self.xPrior)
		self.x = self.xPrior + dot(K, v1)
		# Update state covariance
		M3 = identity(size(self.x)) - dot(K, self.H)
		self.P = dot(M3, self.PPrior)
		
	def getCurrentState(self):
		return self.x