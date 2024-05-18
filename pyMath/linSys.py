import numpy as np
import copy
import sys


"""
/********************/
係数行列の対角要素が非ゼロで上三角な場合の、連立一次方程式の求解
/********************/
input : U, b
	U -> <np:float:(n, n)> 係数行列
	b -> <np:float:(n, )> 右辺ベクトル
output : <np:float:(n, )> 解x
"""
def Utri(U, b):
	n = len(b)
	x = np.zeros(n)

	x[-1] = b[-1]/U[-1,-1]
	for i in range(n-2, -1, -1):
		x[i] = (b[i] - np.sum(U[i, i+1:]*x[i+1:]))/U[i,i]

	return x


"""
/********************/
ガウスの消去法による連立一次方程式求解
/********************/
input : _A, _b, pivot
	_A -> <np:float:(n, n)> 係数行列
	_b -> <np:float:(n, )> 右辺ベクトル
	pivot -> <bool> pivot操作の要否
output : <np:float:(n, )> 解x
Note:
---対称正定値行列---
対称正定値行列の場合、pivot処理を行わなくても、計算の破綻は生じない。
"""
def GaussElim(_A, _b, pivot = True):
	A = copy.deepcopy(_A) #係数行列の要素を書き換えるので、コピーを作成。
	b = copy.deepcopy(_b)
	n = len(b) #次元数

	fancy_index = np.arange(n).astype(int) #ピボット処理のログ

	for i in range(n-1):

		if pivot:
			#####対角要素の選定、ピボット処理
			pivot_index = np.argmax(np.abs(A[i]))
			fancy_index[i] = pivot_index
			fancy_index[pivot_index] = i
			A = A[:,fancy_index]

		#####上三角化
		b[i+1:] -= A[i+1:,i]*b[i]/A[i,i]
		for j in range(i+1,n):
			A[j,i:] -= A[j,i]*A[i,i:]/A[i,i]

	x = Utri(A, b)

	return x[fancy_index]

"""
/********************/
LU分解を利用した、連立一次方程式の求解
/********************/
input : _A, _b
	_A -> <np:float:(n, n)> 係数行列
	_b -> <np:float:(n, )> 右辺ベクトル
	LI -> <np:float:(n, n)> LI = L1L2... "数値線形代数の数理とHPC, p8, 式(1.8)"
	U -> <np:float:(n, n)> 上三角行列
	_return -> <bool> 解析結果としてLI, U, fancy_indexも返すか否か
output : <np:float:(n, )> 解x
"""
def linSysLU(_b, _A = None, pivot = True, LI = None, U = None, _return = True):
	n = len(b) #次元数

	if U is None:
		U = copy.deepcopy(_A) #係数行列の要素を書き換えるので、コピーを作成。
		b = copy.deepcopy(_b)
		LI = np.eye(n)

		fancy_index = np.arange(n).astype(int) #ピボット処理のログ

		for i in range(n-1):
			if pivot:
				#####対角要素の選定、ピボット処理
				pivot_index = np.argmax(np.abs(A[i]))
				fancy_index[i] = pivot_index
				fancy_index[pivot_index] = i
				A = A[:,fancy_index]

			#####上三角化
			b[i+1:] -= U[i+1:,i]*b[i]/U[i,i]
			for j in range(i+1,n):
				LI[j,i] = -U[j,i]/U[i,i]
				U[j,i:] -= U[j,i]*U[i,i:]/U[i,i]

		x = Utri(U, b)
		if _return:
			return x[fancy_index], U, LI, fancy_index
		else:
			return x[fancy_index]

	else:
		b = copy.deepcopy(_b)

		for i in range(n-1):
			Li = np.eye(n)
			Li[i+1:,i] = LI[i+1:,i]
			b = Li@b

		x = Utri(U, b)

		return x