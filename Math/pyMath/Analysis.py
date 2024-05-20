'''
/******************************************/
module : pyMath version 1.0
File name : Analysis.py
Author : Yuji Nakanishi
Latest update : 2019/12/22
/******************************************/
	解析学に関する関数を格納。

Class list:
Function list:
	Gradient, 
'''

'''
/**************/
Gradient
/**************/
type : function
process : gradientを計算
Input : function, x, h
	function -> func。
		Input -> numpy配列。(D, )なshape
		Output -> float．
	x -> gradientを計算する点
	h -> float。微小差。
Output : grad．gradient．(D, )なshape
Note
	:中心差分．
'''
def Gradient(function, x, h = 1e-5):
	import numpy as np

	D = x.shape[0] #次元数

	grad = None #最終的にはnumpy配列になる。

	#####ステップ値を用意
	step = np.eye(D)*h

	for s in step:
		#sは(D, )なshape
		##########itr(反復回数)次元目の微分値を計算
		x_plus = x + s; x_minus = x - s
		func_plus = function(x_plus); func_minus = function(x_minus)

		if grad is None:
			grad = np.array([(func_plus - func_minus)/(2.*h)])
		else:
			grad = np.concatenate((grad, np.array([(func_plus - func_minus)/(2.*h)])))

	return grad

'''
/**************/
HessianMatrix
/**************/
type : function
process : ヘッセ行列を計算を計算
Input : function, X, h
	function -> func。
		Input -> numpy配列。(D, )なshape
		Output -> float．
	X -> gradientを計算する点
	h -> float。微小差。
Output : ヘッセ行列．(D, D)なshape
Note
	:gradientの計算は中心差分．
	:二階微分の計算は全身差分．
'''
def HessianMatrix(X, function, h = 1e-5):
	import numpy as np
	
	D = X.shape[0] #次元数
	H = np.zeros((D, D)) #ヘッセ行列．

	gradient = Gradient(function, X, h) #gradient．(D, )なshape

	##########ヘッセ行列の要素を計算．
	#####i行目の計算．
	for i in range(D):
		#####前進差分を行うために，x+hのベクトルが必要．
		X_pare = X.copy()
		#####要素iのみ+h
		X_pare[i] += h

		#####X_pareでのgradientを計算
		gradient_pare = Gradient(function, X_pare, h) #(D, )なshape

		#####2回微分を計算．前進差分．
		gradient_2 = (gradient_pare - gradient) / h #(D, なshape)

		#####H[i, j]及びH[j, i]に値を代入．
		for j in range(i, D):
			H[i, j] = gradient_2[j]; H[j, i] = gradient_2[j]

	return H