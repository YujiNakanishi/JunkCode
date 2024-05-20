import numpy as np

class epsilon_greedy:
	def __init__(self, epsilon, arms):
		self.epsilon = epsilon
		self.arms = arms
		self.ns = np.zeros(self.arms)
		self.Q = np.zeros(self.arms)

	def policy(self):
		greedy_flag = False if (self.epsilon > np.random.rand()) else True

		act_candidate = np.where(self.Q == np.max(self.Q))[0] \
		if greedy_flag else np.arange(self.arms)
		act = np.random.choice(act_candidate)

		return act

	def study(self, act, r):
		self.ns[act] += 1
		self.Q[act] = (1.-1./self.ns[act])*self.Q[act] + r/self.ns[act]

class softmax(epsilon_greedy):
	def __init__(self, tau, arms):
		self.arms = arms
		self.tau = tau
		self.ns = np.zeros(self.arms)
		self.Q = np.zeros(self.arms)

	def policy(self):
		prob = np.exp(self.Q/self.tau)
		prob /= np.sum(prob)

		act = np.random.choice(np.arange(self.arms), p = prob)
		return act


class epsilon_greedy_const(epsilon_greedy):
	def __init__(self, epsilon, arms, alpha):
		super().__init__(epsilon, arms)
		self.alpha = alpha

	def study(self, act, r):
		self.ns[act] += 1
		self.Q[act] = (1.-self.alpha)*self.Q[act] + self.alpha*r