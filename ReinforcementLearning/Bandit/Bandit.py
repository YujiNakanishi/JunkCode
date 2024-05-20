import numpy as np


class bandit:
	def __init__(self, levers):
		self.levers = levers
		self.rate = np.random.rand(self.levers)

	def play(self, lever):
		return 1 if (self.rate[lever] > np.random.rand()) else 0

class ustdbandit(bandit):
	def play(self, arm):
		self.rate += 0.1*np.random.randn(self.arms)
		return 1 if (self.rate[arm] > np.random.rand()) else 0