import numpy as np
import pprint

class LogBarrier(object):
	"""
		C: 		n * 1
		Q: 		n * n
		alpha : n * 1
		h : 	n * 1
	"""
	def __init__(self):
		pass

	def init_alpha(self, Q, C, epsilon):
		n = Q.shape[0]
		alpha_est = np.matmul(np.linalg.inv(Q), (np.ones((n, 1)) - C))
		l_alpha_pos = [alpha_est[j][0] if alpha_est[j][0] > 0 else epsilon for j in range(n)]
		return np.array(l_alpha_pos).reshape(n, 1)

	def compute_alpha(self, Q, C):
		epsilon = 1.e-3
		r = 0.99

		n = Q.shape[0]
		# alpha = np.ones((n, 1)) * 10.0 #self.init_alpha(Q, C, epsilon)
		alpha = self.init_alpha(Q, C, epsilon)
		print (alpha)

		h = np.matmul(Q, alpha) + C - 1.0/(n *alpha)
		converge_cond = np.linalg.norm(h) <= epsilon

		count = 0
		while not converge_cond and count < 10000:
			d = self.Newtons_method(Q, alpha, h)
			selected_d_over_delta_d = [alpha[j] / -d[j] for j in range(n) if d[j] < 0]
			print(len(selected_d_over_delta_d))
			theta = min(1.0, r * min(selected_d_over_delta_d)) if selected_d_over_delta_d else 1.0
			# theta = 1.0
			alpha = alpha + theta * d
			h = np.matmul(Q, alpha) + C - 1.0/(n *alpha)
			# h = np.matmul(Q, alpha) + C
			converge_cond = np.linalg.norm(h) <= epsilon
			count += 1

			import time
			print (np.linalg.norm(theta))
			print (np.linalg.norm(d))
			print (np.linalg.norm(1.0/(n *alpha)))
			print (np.linalg.norm(h))
			print ("       ")
			time.sleep(0.5)

		print (alpha)
		return alpha

	def Newtons_method(self, Q, alpha, h):
		
		n = alpha.shape[0]
		# 1
		print(alpha[0])
		dB = np.array(list(map(lambda x: 1.0 / (x**2), alpha)))
		# print(dB.shape)
		delta = np.diag(dB[:,0])
		# print("drtestesras: ",delta.shape)
		H = Q + 1.0 / n * delta
		H_inv = np.linalg.inv(H)
		print(np.linalg.det(H), np.linalg.det(H_inv))
		res = -np.matmul(H_inv, h)
		return res



if __name__ == '__main__':
	dim = 100
	Q = np.random.rand(dim, dim)
	Q = Q.T @ Q
	print(np.linalg.det(Q))
	C = np.random.rand(dim, 1)
	l = LogBarrier()
	l.compute_alpha(Q, C)


