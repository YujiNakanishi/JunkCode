import numpy as np
import copy
import sys

def norm(x):
	return np.sqrt(np.sum(np.abs(x)**2))


"""
/***********************/
Bi-CGSTAB法変形版1による連立一次方程式求解
/***********************/
input:
	field -> <class:field>
	iteration -> <int> 最大反復回数
	epsilon -> <float> 最大許容残差
output:
	log -> <np:float:(itr, )> 残差の系列
"""
def BiCGSTABver1(field, iteration, epsilon):
	tol_S = norm(field.S)
	r = field.S - field.mat(field.P)
	u = copy.deepcopy(r)

	rs = 2.*(np.random.rand(field.N)-0.5) + 2.j*(np.random.rand(field.N)-0.5)
	res = norm(field.S - field.mat(field.P))
	log = []

	for itr in range(iteration):
		log.append(res)
		c = field.mat(u)
		sigma = np.vdot(rs, c)
		alpha = np.vdot(rs, r)/sigma
		r -= alpha*c
		field.P += alpha*u
		s = field.mat(r)
		beta = np.vdot(rs, s)/sigma
		c = s - beta*c
		u = r - beta*u
		eta = np.vdot(s, r)/np.vdot(s, s)
		field.P += eta*r
		r -= eta*s
		u -= eta*c
		res = norm(field.S - field.mat(field.P))
		if res/tol_S < epsilon:
			break

	log.append(res)

	return log