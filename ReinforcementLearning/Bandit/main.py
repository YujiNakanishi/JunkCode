import numpy as np
import Bandit
import Agent
import pandas as pd

arms = 100
time_step = 1000
alpha = 0.8
epsilon = 0.1
trial_num = 100

def trial():
	bandit = Bandit.ustdbandit(arms)
	agent = Agent.epsilon_greedy(epsilon, arms)

	Reward = np.zeros(time_step)

	for t in range(time_step):
		act = agent.policy()
		Reward[t] = bandit.play(act)
		agent.study(act, Reward[t])

	wining_rate = np.zeros(time_step)
	for t in range(1, time_step+1):
		wining_rate[t-1] = np.sum(Reward[:t])/t

	return wining_rate

Wining_rate = np.zeros(time_step)
for t in range(trial_num):
	Wining_rate += trial()
Wining_rate /= trial_num


data = np.stack((np.arange(1, time_step+1), Wining_rate), axis = 1)
data = pd.DataFrame(data)
data.to_csv("data.csv")