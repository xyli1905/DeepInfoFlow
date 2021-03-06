import numpy as np
# from scipy import optimize
import pprint
import time

class LogBarrier(object):
	"""
		C: 		n * 1
		Q: 		n * n
		alpha : n * 1
		h : 	n * 1
	"""
	def __init__(self):
		# params
		self.accuracy = 1.e-16 # avoiding numerical overflow
		self.MAX_COUNT = 10000 # max iter number in compute_alpha
		self.epsilon = 1.e-10  # control convergence of compute_alpha

		if __name__ == '__main__':
			self.PRINT = True
		else:
			self.PRINT = False

	def _init_alpha(self, Q, C):
		n = Q.shape[0]
		try:
			tmp = np.diag(np.ones(n))
			alpha_est = np.matmul(np.linalg.inv(Q + tmp), (np.ones((n, 1)) - C))
			l_alpha_pos = [alpha_est[j][0] if alpha_est[j][0] > 0 else self.accuracy for j in range(n)]
		except Exception as e:
			print("gua in init_alpha")
			print(e)
			f = open("tmp_init.txt", "w")
			f.write(str(list(Q)))
			f.close()
			return np.ones((n,1)) * 1e-3 # avoid numerical error in inv(Q + tmp)
		return np.array(l_alpha_pos).reshape(n, 1)

	def compute_alpha(self, Q, C):
		# params
		r = 0.99 # can be any number that restrictly < 1.0
		n = Q.shape[0]

		# initialize alpha for iteration
		alpha = self._init_alpha(Q, C)

		h = np.matmul(Q, alpha) + C - 1.0/(n * alpha + self.accuracy)
		converge_cond = np.linalg.norm(h) <= self.epsilon

		# main loop
		count = 0
		last_d = np.ones_like(h) * 1e-3 # avoid svd error in pinv() at first update
		while not converge_cond and count < self.MAX_COUNT:
			indicator, d = self._Newtons_method(Q, alpha, h)
			if indicator:
				last_d = d
			else:
				d = last_d
			selected_d_over_delta_d = [alpha[j] / (-d[j] + self.accuracy) for j in range(n) if d[j] < 0]
			theta = min(1.0, r * min(selected_d_over_delta_d)) if selected_d_over_delta_d else 1.0
			alpha = alpha + theta * d
			h = np.matmul(Q, alpha) + C - 1.0/(n * alpha + self.accuracy)
			converge_cond = np.linalg.norm(h) <= self.epsilon
			count += 1

			#Debug outputs
			# import time
			# print (np.linalg.norm(theta))
			# print (np.linalg.norm(d))
			# print (np.linalg.norm(1.0/(n *alpha)))
			# print (np.linalg.norm(h))
			# print ("       ")
			# time.sleep(0.5)

		# output
		if count >= self.MAX_COUNT:
			print("Terminated (maximum number of iterations reached).")
		if self.PRINT:
			print("After converge(termination), the val of ||h||_2 and f, i.e. the objective, are:")
			print("\t||h||_2 = ", np.linalg.norm(np.matmul(Q, alpha) + C - 1.0/(n * alpha)))
			print("\tf = ", 0.5 * alpha.T @ Q @ alpha + C.T @ alpha - np.sum(np.log(alpha))/n)

		return alpha

	def _Newtons_method(self, Q, alpha, h):
		'''
		Newton's method, getting the update step
		'''
		n = alpha.shape[0]
		dB = np.array(list(map(lambda x: 1.0 / (x**2 + self.accuracy), alpha)))
		delta = np.diag(dB[:,0])
		H = Q + 1.0 / n * delta
		H = np.nan_to_num(H) #avoid +-inf, making pinv more stable
		try:
			H_inv = np.linalg.pinv(H)
			res = -np.matmul(H_inv, h)
		except Exception as e:
			print("gua in newton method")
			print(e)
			f = open("tmp_newton.txt", "w")
			f.write(str(list(H)))
			f.close()
			return False, np.zeros_like(h)
		return True, res



if __name__ == '__main__':
	dim = 500

	#test samples
	# x = np.random.rand(dim, 16)
	# y = np.random.rand(dim, 16)
	# x = np.random.normal(4., 2., (dim, 1))
	# y = np.random.normal(0., 1., (dim, 1))
	x = np.random.normal(0., 1., (dim, 1))
	y = np.random.normal(4., 2., (dim, 1))
	# x = np.random.beta(1., 2., size=(dim, 1))
	# y = np.random.uniform(low=0.0, high=1.0, size=(dim, 1))

	lambdaN = 1./dim
	sigma = .1

	# compute Kyy
	var = np.zeros((dim, dim))
	for i in range(dim):
		for j in range(dim):
			delta = x[i,:] - y[j,:]
			mu2 = np.dot(delta, delta)
			var[i,j] = -1.0 * mu2 / sigma
	Kxy = np.exp(var)
	c_n = np.transpose(Kxy) @ np.ones((dim, 1)) / (-1.0 * lambdaN * dim)
	# print(c_n.shape)

	# compute Kyy
	var = np.zeros((dim, dim))
	for i in range(dim):
		for j in range(i, dim):
			delta = y[i,:] - y[j,:]
			mu2 = np.dot(delta, delta)
			var[i,j] = -1.0 * mu2  / sigma
			var[j,i] = var[i,j]
	Kyy = np.exp(var)
	Q_n = Kyy / lambdaN
	# print(np.linalg.det(Q_n))

	t_begin = time.time()
	l = LogBarrier()
	alpha = l.compute_alpha(Q_n, c_n)
	print("\tDhat_kl = ", -np.log(dim) - np.sum(np.log(alpha))/dim)
	t_end = time.time()
	print(f"time cost {t_end - t_begin:.5f}(s)")
