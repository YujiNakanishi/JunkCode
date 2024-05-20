import numpy as np
import sys

"""
process : 単純な探査手法。N回の計算コスト。
input : x, X, n, L, function = None, oput, gpu
	x -> <np array> (D, )なshape
	X -> <np array> (N, D)なshape
	n -> <int> 近傍数
	L -> <float> 近傍範囲
	function -> <func> 距離測度。if None -> ユークリッド距離
	oput -> <str> 出力形式
	gpu -> <bool> cupyを使うか否か
output : <np array> 
oput == "index" -> 近傍点のインデックス値。(a, )なshape。 *aは近傍点の数
oput == "value" -> 近傍点の数値。(a, D)なshape。
oput == "both" -> 両方。
note :
---関数について---
xに対して近傍の点をXから抽出。
---n, Lについて---
nかLのうち少なくとも一方は!(None)でなければならない。どちらもNoneの場合、全点が近傍として返ってくる。
---functionについて---
・input : x, X
・output : <np array> (N, )なshape
---数値実験結果---
3次元空間に対し、len(X)が100000を超えたあたりでGPU計算の方が速くなった。
"""
def SIMPLE(x, X, n = None, L = None, function = None, oput = "both", gpu = False):
	if gpu:
		import cupy as xp
	else:
		import numpy as xp

	if function is None:
		function = lambda x, X : xp.linalg.norm(X-x, ord=2, axis = -1)

	candidate = xp.arange(len(X))
	distance = function(x, X) #(N, )。距離

	if not(L is None):
		candidate = xp.where((distance <= L))[0]
		distance = distance[candidate]

	if not(n is None):
		arg_min = xp.argsort(distance)[:n]
		if oput == "index":
			return candidate[arg_min]
		elif oput == "value":
			return X[arg_min]
		else:
			return candidate[arg_min], X[arg_min]

	else:
		if oput == "index":
			return candidate
		elif oput == "value":
			return X[candidate]
		else:
			return candidate, X[candidate]


"""
/***************/
process : bucket法による近傍探査の管理
/***************/
---field---
X -> <np array> データセット。(N, D)なshape
mode -> <str> "numpy" or "cupy"
X_min -> <np array> 各軸の最小値。(D, )なshape
bucket -> <np array> 各点のバケット番号
---Note---
<近傍点の定義>
自身のバケットと隣接するバケットに属する点すべて
"""
class Bucket:
	def __init__(self, X, h, mode = "numpy"):
		self.mode = mode
		self.update(X, h)

	def to(mode):
		import cupy as cp
		if (mode == "numpy") & (self.mode == "cupy"):
			self.mode = "numpy"
			self.X = cp.asnumpy(self.X)
			self.bucket = cp.asnumpy(self.bucket)
		elif (mode == "cupy") & (self.mode == "numpy"):
			self.mode = "cupy"
			self.X = cp.array(self.X)
			self.bucket = cp.array(self.bucket)

	def update(self, X, h):
		if self.mode == "numpy":
			import numpy as xp
		else:
			import cupy as xp
		
		self.X = X
		self.h = h
		self.X_min = xp.array([xp.min(self.X[:,d]) for d in range(self.X.shape[1])])


		X_shift = self.X - self.X_min

		self.bucket = [xp.floor(X_shift[:,d]/self.h).astype("int") for d in range(self.X.shape[1])]
		self.bucket = xp.stack(self.bucket, axis = -1)

	def neighber(self, x, oput = "both"):
		if self.mode == "numpy":
			import numpy as xp
		else:
			import cupy as xp

		x_shift = x - self.X_min

		buc = [xp.floor(xx/self.h).astype("int") for xx in x_shift]
		mask = xp.ones(self.X.shape[0]).astype("bool")
		for d in range(self.X.shape[1]):
			mask *= (xp.abs(self.bucket[:,d]-buc[d])<=1.)

		candidate = xp.where(mask)[0] 
		if oput == "index":
			return candidate
		elif oput == "value":
			return self.X[candidate]
		else:
			return candidate, self.X[candidate]