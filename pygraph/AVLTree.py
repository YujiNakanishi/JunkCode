"""
/****************/
AVLTree
/****************/
class and function for AVLtree is defined.
"""

import pygraph

"""
/***************/
AVLtree
/***************/
Field  : nodes -> list of dictionary.

Method : 
<inherited from Base> __getitem__, __len__, list, copy, equal, greater
<inherited from rootedtree> isConnect, Root, Degree, Leaf, Sibling, Height, Depth, isAncestor, isDescendant, 
                            Descendants, Ancestors, isLeaf, isRoot
<defined from binarytree> child, reindex, Subtree
<defined from binarysearchtree> __init__, max, argmax, min, argmin, index
<defined in this class> append, remove, getBalance, rotateLeft, rotateRight, balanced
Note   :
--- Introduction ---
This is for binarysearchtree graph. Greater nodes are on the right side.

--- Node ---
node has four keys of "value", "parent", "left" and "right".
 "value" is as explained if Base class.
 "parent" is int or None. parent node index. if None, this node is root.
 "left" and "right" is int or None. left(right) node index.
"""
class AVLtree(pygraph.BinarySearchTree.binarysearchtree):

	"""
	Process : make tree balanced
	"""
	def balanced(self, unbalance_index, v):
		while True:
			if unbalance_index is None:
				break
			else:
				unbalance = self.getBalance(unbalance_index)

				if (-2 < unbalance < 2):
					unbalance_index = self[unbalance_index]["parent"]
				else:
					break

		if not(unbalance_index is None):
			if (self.greater(self[unbalance_index]["value"], v)):
				left_index = self[unbalance_index]["left"]

				if (self.greater(self[left_index]["value"], v)):
					self.rotateRight(unbalance_index)
				else:
					p_index = self[unbalance_index]["parent"]
					self.rotateLeft(unbalance_index)
					self.rotateRight(p_index)

			else:
				right_index = self[unbalance_index]["right"]

				if (self.greater(v, self[right_index]["value"])):
					self.rotateLeft(unbalance_index)
				else:
					p_index = self[unbalance_index]["parent"]
					self.rotateRight(unbalance_index)
					self.rotateLeft(p_index)


	def append(self, v):
		super().append(v)

		#####check unbalance node
		unbalance_index = self[-1]["parent"]
		self.balanced(unbalance_index, v)

	"""
	Process : remove node i
	Note : This method can be used only if node i is a leaf.
	"""
	def remove(self, i):
		unbalance_index = self[i]["parent"]
		if unbalance_index > i:
			unbalance_index -= 1
		super().remove(i)

		balance = self.getBalance(unbalance_index)

		if balance > 1:
			self.rotateLeft(unbalance_index)
		elif balance < -1:
			self.rotateRight(unbalance_index)


	def getBalance(self, idx):
		right = self[idx]["right"]
		left = self[idx]["left"]

		if right is None:
			right_height = -1
		else:
			right_height = self.Height(right)

		if left is None:
			left_height = -1
		else:
			left_height = self.Height(left)

		return left_height - right_height

	"""
	process : rotate right toward index i
	"""
	def rotateLeft(self, i):
		if not(self[i]["right"] is None):
			i_parent = self[i]["parent"]
			i_left = self[i]["left"]
			i_right = self[i]["right"]

			ir_left = self[i_right]["left"]

			self[i_right]["parent"] = i_parent
			if not(i_parent is None):
				self[i_parent]["right"] = i_right

			self[i]["parent"] = i_right
			self[i_right]["left"] = i

			self[i]["right"] = ir_left
			if not(ir_left is None):
				self[ir_left]["parent"] = i


	def rotateRight(self, i):
		if not(self[i]["left"] is None):
			i_parent = self[i]["parent"]
			i_left = self[i]["left"]
			i_right = self[i]["right"]

			il_right = self[i_left]["right"]

			self[i_left]["parent"] = i_parent
			if not(i_parent is None):
				self[i_parent]["left"] = i_left

			self[i]["parent"] = i_left
			self[i_left]["right"] = i

			self[i]["left"] = il_right
			if not(il_right is None):
				self[il_right]["parent"] = i