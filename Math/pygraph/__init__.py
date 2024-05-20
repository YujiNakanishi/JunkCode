"""
/****************/
pygraph ver. 1.4
/****************/
Author : Yuji Nakanishi
2021/04/25

class : 
 Base
Function :
 AdjacentMatrix 
"""

import numpy as np
import copy as cpy

"""
/**************/
Base
/**************/
Field : nodes -> list of dictionary.
Note : 
--- Introduction ---
This class is an ancester for any classes in pygraph.
Any classes have nodes field. nodes is a list of dictionary, and this dictionary has information about nodes.
 The information nodes must have depends on data structure. Therefore, nodes can't be defined in Base class.
However, since node in any data structure should have value which can be any type, dictionary wwhich represents
node information of any data structure shoud have key "value".

--- Comaprision ---
The comparision between two nodes is often required. While it is easy in case that node["value"] is number,
definition of quality should be decided in other cases.
 In Base class, there are method "equal" and "greater" for comparision which is assumed "value" is number, and
any classes inherit this. Therefore, if "value" isn't number, you need to define a class explicitly and overwrite
both method.

"""
class Base:

	def __getitem__(self, i):
		return self.nodes[i]

	def __len__(self):
		return len(self.nodes)

	"""
	process : return list of values which is sorted by index
	output  : list of value
	"""
	def list(self):
		return [node["value"] for node in self.nodes]

	"""
	process : return a copy of data structure
	output  : class of data structure
	"""
	def copy(self):
		nodes = [cpy.deepcopy(n) for n in self.nodes]
		return type(self)(nodes = nodes)

	"""
	process : compare between node values v and w.
	input: v, w -> any type. node["value"]
	output  : boolean
	Note : It is assumed node["value"] is number.
	       If it's incorrect, please define class explicitly and overwrite this.
	"""
	def  equal(self, v, w):
		return v == w

	"""
	process : compare between node values v and w.
	input   : v, w -> any type. node["value"]
	output  : boolean
	Note    : It is assumed node["value"] is number.
	          If it's incorrect, please define class explicitly and overwrite this.
	"""
	def greater(self, v, w):
		return (v > w)

	"""
	process : search node index which value is v
	input   : v, only
		v -> any type. node["value"]
		only -> boolean.
			If True, return just one index which is found first.
			If False, return all indice.
	output  : int(only == True) or list(only == False) of int
	Note    : search algorithm is list base.
	"""
	def index(self, v, only = False):
		if only:
			##### define list of node values
			values = self.list()

			for idx, value in enumerate(values): ### for all value
				if self.equal(value, v): ###if v == value
					return idx
		else:
			return [idx for idx, value in enumerate(self.list()) if self.equal(value, v)]


from pygraph import Undigraph
from pygraph.Undigraph import undigraph
from pygraph import Digraph
from pygraph.Digraph import digraph
from pygraph import RootedTree
from pygraph.RootedTree import rootedtree
from pygraph import BinaryTree
from pygraph.BinaryTree import binarytree
from pygraph import BinarySearchTree
from pygraph.BinarySearchTree import binarysearchtree
from pygraph import AVLTree
from pygraph.AVLTree import AVLtree
from pygraph import Heap
from pygraph.Heap import maxheap, minheap

"""
process : return adjacent matrix
input   : graph -> class of graph
output  : matrix -> np array. shape is (D, D) where D = len(graph)
"""
def AdjacentMatrix(graph):
	#####define void matrix.
	size = len(graph)
	matrix = np.zeros((size, size)).astype(int)

	if type(graph) == undigraph:
		for idx, n in enumerate(graph[:int(size/2)+1]):
			for a in n["adjacent"]:
				matrix[idx, a] = 1
				matrix[a, idx] = 1

	elif type(graph) == digraph:
		for idx, n in enumerate(graph):
			for a in n["adjacent"]:
				matrix[idx, a] = 1

	elif type(graph) in [rootedtree, binarytree, maxheap, minheap, binarysearchtree, AVLtree]:
		for idx, node in enumerate(graph):
			p = node["parent"]
			if not(p is None):
				matrix[idx, p] = 1
				matrix[p, idx] = 1

	return matrix