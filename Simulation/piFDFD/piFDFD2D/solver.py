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

	rs = 2.*(np.random.rand(field.shape[0], field.shape[1])-0.5) + 2.j*(np.random.rand(field.shape[0], field.shape[1])-0.5)
	res = norm(field.S[field.nanMask == False] - field.mat(field.P)[field.nanMask == False])
	log = []

	for itr in range(iteration):
		log.append(res)
		print(str(itr)+"\t"+str(log[-1]))
		c = field.mat(u)
		sigma = np.vdot(rs[field.nanMask == False], c[field.nanMask == False])
		alpha = np.vdot(rs[field.nanMask == False], r[field.nanMask == False])/sigma
		r -= alpha*c
		field.P += alpha*u

		s = field.mat(r)
		beta = np.vdot(rs[field.nanMask == False], s[field.nanMask == False])/sigma
		c = s - beta*c
		u = r - beta*u

		eta = np.vdot(s[field.nanMask == False], r[field.nanMask == False])/np.vdot(s[field.nanMask == False], s[field.nanMask == False])
		field.P += eta*r
		r -= eta*s
		u -= eta*c

		res = norm(field.S[field.nanMask == False] - field.mat(field.P)[field.nanMask == False])
		if res/tol_S < epsilon:
			break

	log.append(res)

	return log


"""
/***********************/
IDR(s)法による連立一次方程式求解
/***********************/
input:
	field -> <class:field>
	iteration -> <int> 最大反復回数
	epsilon -> <float> 最大許容残差
output:
	log -> <np:float:(itr, )> 残差の系列
"""
def IDR_s(field, s, iteration = 1000, epsilon = 1e-10):
	tol_S = norm(field.S)
	Ro = [2.*(np.random.rand(field.shape[0], field.shape[1])-0.5) + 2.j*(np.random.rand(field.shape[0], field.shape[1])-0.5) for i in range(s)]
	Ro = [Ro[i]/norm(Ro[i]) for i in range(s)]
	for i in range(1, s):
		for j in range(i):
			rv = np.sum(Ro[i]*Ro[j])
			Ro[i] -= rv*Ro[j]
		Ro[i] /= norm(Ro[i])
	Ro = np.stack(Ro, axis = -1) #<np:float(n, s)>

	r = field.S - field.mat(field.P)
	U = []; S = []
	for i in range(s):
		c = field.mat(r)
		w = np.vdot(c[field.nanMask == False], r[field.nanMask == False])/np.vdot(c[field.nanMask == False], c[field.nanMask == False])
		U.append(w*r)
		S.append(-w*c)

		field.P += U[-1]
		r += S[-1]

	U = np.stack(U, axis = -1)
	S = np.stack(S, axis = -1)


	sigma = (np.conjugate(Ro[field.nanMask == False].T).reshape((s, -1)))@(S[field.nanMask == False].reshape((-1, s)))
	rho = (np.conjugate(Ro[field.nanMask == False].T).reshape((s, -1)))@(r[field.nanMask == False].reshape((-1,1)))[:,0]
	j = 1

	res = norm(field.S[field.nanMask == False] - field.mat(field.P)[field.nanMask == False])
	log = []
	for itr in range(iteration):
		log.append(res)
		print(str(itr)+"\t"+str(log[-1]))
		for k in range(s):
			gamma = np.linalg.solve(sigma, rho) #<np:complex:(s, )>
			q = -S@gamma
			v = r + q

			if k == 0:
				c = field.mat(v)
				w = np.vdot(c[field.nanMask == False], v[field.nanMask == False])/np.vdot(c[field.nanMask == False], c[field.nanMask == False])
				S[:,:,j] = q - w*c
				U[:,:,j] = -U@gamma + w*v
			else:
				U[:,:,j] = -U@gamma + w*v
				S[:,:,j] = -field.mat(U[:,:,j])

			r += S[:,:,j]
			field.P += U[:,:,j]

			dm = (np.conjugate(Ro[field.nanMask == False].T).reshape((s, -1)))@(S[:,:,j][field.nanMask == False].reshape((-1, 1)))[:,0]

			sigma[:,j] = dm
			rho += dm
			j += 1
			if j >= s:
				j = 1

		res = norm(field.S[field.nanMask == False] - field.mat(field.P)[field.nanMask == False])
		if res/tol_S < epsilon:
			break

	log.append(res)

	return log