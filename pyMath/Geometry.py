'''
/******************************************/
module : pyMath version 1.0
File name : Geometry.py
Author : Yuji Nakanishi
Latest update : 2019/12/22
/******************************************/
	幾何学に関する関数を格納。

Class list:
Function list:
	Rotate
'''

import math

'''
/*****************/
Rotate
/*****************/
type : function
process : 点の回転移動
Input : origin, axis, X, theta, use_gpu
	origin -> xp配列．(3, )なshape．回転軸が通る点
	axis -> xp配列．(3, )なshape．回転軸
	X -> (N, 3)なshape．点群．
	theta -> float．回転角度(rad)．0 <=> 2pi
	use_gpu -> boolen．trueならcupyで計算．
Output : X
'''
def Rotate(X, axis, theta, origin = None, use_gpu = False):
	if use_gpu:
		import cupy as xp
	else:
		import numpy as xp

	if origin is None:
		origin = xp.array([0.0, 0.0, 0.0]) #原点座標にする．
	
	#####平行移動
	X -= origin

	#####axisを単位ベクトルにする．
	axis /= xp.sqrt(axis[0]**2+axis[1]**2+axis[2]**2)

	R = xp.array([
	[axis[0]**2+(1-axis[0]**2)*xp.cos(theta), axis[0]*axis[1]*(1-xp.cos(theta))+axis[2]*xp.sin(theta), axis[0]*axis[2]*(1-xp.cos(theta))-axis[2]*xp.sin(theta)],
	[axis[0]*axis[1]*(1-xp.cos(theta))-axis[2]*xp.sin(theta), axis[1]**2+(1-axis[1]**2)*xp.cos(theta), axis[1]*axis[2]*(1-xp.cos(theta))+axis[2]*xp.sin(theta)],
	[axis[0]*axis[2]*(1-xp.cos(theta))+axis[1]*xp.sin(theta), axis[1]*axis[2]*(1-xp.cos(theta))-axis[0]*xp.sin(theta), axis[2]**2+(1-axis[2]**2)*xp.cos(theta)]
	])

	X = xp.dot(R, X.T).T + origin

	return X