"""
/****************/
BinarySearchTree
/****************/
class and function for binarysearchtree is defined.
"""

import pygraph

"""
/***************/
binarysearchtree
/***************/
Field  : nodes -> list of dictionary.

Method : 
<inherited from Base> __getitem__, __len__, list, copy, equal, greater
<inherited from rootedtree> isConnect, Root, Degree, Leaf, Sibling, Height, Depth, isAncestor, isDescendant, 
                            Descendants, Ancestors, isLeaf, isRoot
<defined from binarytree> child, reindex, Subtree, remove
<defined in this class> __init__, append, max, argmax, min, argmin, index
Note   :
--- Introduction ---
This is for binarysearchtree graph. Greater nodes are on the right side.

--- Node ---
node has four keys of "value", "parent", "left" and "right".
 "value" is as explained if Base class.
 "parent" is int or None. parent node index. if None, this node is root.
 "left" and "right" is int or None. left(right) node index.
"""
class binarysearchtree(pygraph.BinaryTree.binarytree):

	def __init__(self, V = None, nodes = None):
		if nodes is None:
			self.nodes = [{"value" : V[0], "left" : None, "right" : None, "parent" : None}]

			for v in V[1:]:
				self.append(v)
		else:
			self.nodes = nodes

	"""
	Process : append new node
	input : v, current_index
		v -> any type of value
		current_index -> int. root index. If you don't know previously, set None
	"""
	def append(self, v, current_index = None):
		new_index = len(self) #index of a new node
		if current_index is None:
			current_index = self.Root(0)

		while True:
			if self.greater(v, self[current_index]["value"]):
				if self[current_index]["right"] is None:
					self.nodes.append({"value" : v, "left" : None, "right" : None, "parent" : current_index})
					self[current_index]["right"] = new_index
					break
				else:
					current_index = self[current_index]["right"]
			
			else:
				if self[current_index]["left"] is None:
					self.nodes.append({"value" : v, "left" : None, "right" : None, "parent" : current_index})
					self[current_index]["left"] = new_index
					break
				else:
					current_index = self[current_index]["left"]


	def index(self, v, only = False):
		index = self.Root()

		if only:
			while True:
				if self.equalVal(v, self[index]["value"]):
					return index

				elif self.greaterVal(v, self[index]["value"]):
					index = self[index]["right"]
					if index is None:
						return None
				else:
					index = self[index]["left"]
					if index is None:
						return None
		else:
			indice = []

			while True:
				if self.equalVal(v, self[index]["value"]):
					indice.append(index)
					index = self[index]["left"]
				elif self.greaterVal(v, self[index]["value"]):
					index = self[index]["right"]
				else:
					index = self[index]["left"]

				if index is None:
					return indice


	def max(self):
		index = 0
		while True:
			new_index = self[index]["right"]
			if new_index is None:
				return self[index]["value"]
			index = new_index

	def argmax(self):
		index = 0
		while True:
			new_index = self[index]["right"]
			if new_index is None:
				return index
			index = new_index

	def min(self):
		index = 0
		while True:
			new_index = self[index]["left"]
			if new_index is None:
				return self[index]["value"]
			index = new_index

	def argmin(self):
		index = 0
		while True:
			new_index = self[index]["left"]
			if new_index is None:
				return index
			index = new_index