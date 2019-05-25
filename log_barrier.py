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

	def compute_alpha(self, Q, C, epsilon):

		n = Q.shape[0]
		alpha = np.ones((n, 1)) * 1.0 #self.init_alpha(Q, C, epsilon)
		print (alpha)
		res = np.linalg.norm(np.matmul(Q, alpha) + C + 1.0/(n *alpha))
		converge_cond = res <= epsilon
		r = 0.99
		count = 0
		while not converge_cond and count < 10000:
			d = self.Newtons_method(Q, alpha, h)
			selected_d_over_delta_d = [alpha[j] / -d[j] for j in range(n) if d[j] < 0]
			theta = min(1, r * min(selected_d_over_delta_d)) if selected_d_over_delta_d else 1
			alpha = alpha + theta * d
			count += 1
			import time
			print (np.linalg.norm(np.matmul(Q, alpha)))
			print (np.linalg.norm(C))
			print (np.linalg.norm(1.0/(n *alpha)))
			print (np.linalg.norm(np.matmul(Q, alpha) + C + 1.0/(n *alpha)))
			print ("       ")

			time.sleep(0.5)
			converge_cond = np.linalg.norm(np.matmul(Q, alpha) + C + 1.0/(n *alpha)) <= epsilon
		print (alpha)
		return alpha

	def Newtons_method(self, Q, alpha, h):
		
		n = alpha.shape[0]
		# 1
		alpha = np.array(list(map(lambda x: 1.0 / x**2, alpha)))
		delta = np.diag(alpha)
		H = Q - 1.0 / n * delta
		H_inv = np.linalg.inv(H)
		res = -np.matmul(H_inv, h)
		return res



if __name__ == '__main__':
	dim = 16
	Q = np.random.rand(dim, dim)
	Q = Q.T * Q

	a = np.random.rand(dim, 1)
	h = np.random.rand(dim, 1)
	C = np.random.rand(dim, 1)
	l = LogBarrier()
	l.compute_alpha(Q, C, 0.0001)


