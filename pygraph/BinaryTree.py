"""
/****************/
BinaryTree
/****************/
class and function for binarytree is defined.
"""

import pygraph

"""
/***************/
binarytree
/***************/
Field  : nodes -> list of dictionary.

Method : 
<inherited from Base> __getitem__, __len__, list, copy, equal, greater, index 
<inherited from rootedtree> isConnect, Root, Degree, Leaf, Sibling, Height, Depth, isAncestor, isDescendant, 
                            Descendants, Ancestors, isLeaf, isRoot
<defined in this class> __init__, child, reindex, append, remove, Subtree
Note   :
--- Introduction ---
This is for binarytree graph.

--- Node ---
node has four keys of "value", "parent", "left" and "right".
 "value" is as explained if Base class.
 "parent" is int or None. parent node index. if None, this node is root.
 "left" and "right" is int or None. left(right) node index.
"""
class binarytree(pygraph.RootedTree.rootedtree):

	"""
	input : V, left, right, nodes
		V -> value of list.
		left, right -> node indice. List of int.
		nodes -> list of dictionary.
	"""
	def __init__(self, V = None, left = None, right = None, nodes = None):
		if nodes is None:
			self.nodes = [{"value" : v, "left" : l, "right" : r, "parent" : None} for v, l, r in zip(V, left, right)]

			idx = 0
			for l, r in zip(left, right):
				if not(l is None):
					self[l]["parent"] = idx
				if not(r is None):
					self[r]["parent"] = idx
				idx += 1
		else:
			self.nodes = nodes

	"""
	Process : reindex from root
	"""
	def reindex(self):
		node_index = [self.Root()]
		candidate = self.child(node_index[0])

		while len(node_index) < len(self):
			node_index += [c for c in candidate]
			candidate = sum([self.child(c) for c in candidate], [])

		nodes = [self[n] for n in node_index]

		for idx in range(len(nodes)):
			if not(nodes[idx]["parent"] is None):
				nodes[idx]["parent"] = node_index.index(nodes[idx]["parent"])

			if not(nodes[idx]["left"] is None):
				nodes[idx]["left"] = node_index.index(nodes[idx]["left"])

			if not(nodes[idx]["right"] is None):
				nodes[idx]["right"] = node_index.index(nodes[idx]["right"])

		self.nodes = nodes

	"""
	Process : append new node
	input : v, parent, direction
		v -> any type of value
		parent -> int
		direction -> str. "left" or "right" in which side of parent node is node connected
	Note : This method can be used only if node[parent] is a leaf.
	"""
	def append(self, v, parent, direction):
		new_index = len(self)
		self.nodes.append({"value" : v, "left" : None, "right" : None, "parent" : parent})

		self[parent][direction] = new_index


	"""
	Process : remove node i
	Note : This method can be used only if node i is a leaf.
	"""
	def remove(self, i):
		parent = self[i]["parent"]

		if self[parent]["left"] == i:
			self[parent]["left"] = None
		else:
			self[parent]["right"] = None

		self.nodes.pop(i)

		for idx in range(len(self)):
			if self.isRoot(idx) == False:
				if self[idx]["parent"] > i:
					self[idx]["parent"] -= 1

			if not(self[idx]["left"] is None):
				if self[idx]["left"] > i:
					self[idx]["left"] -= 1

			if not(self[idx]["right"] is None):
				if self[idx]["right"] > i:
					self[idx]["right"] -= 1


	def Subtree(self, root_index):
		index = self.Descendants(root_index) + [root_index]
		V = [self[idx]["value"] for idx in index]

		parents = [index.index(self[idx]["parent"]) for idx in index[:-1]] + [None]
		
		left = []
		right = []
		for idx in index:
			if self[idx]["left"] is None:
				left.append(None)
			else:
				left.append(index.index(self[idx]["left"]))

			if self[idx]["right"] is None:
				right.append(None)
			else:
				right.append(index.index(self[idx]["right"]))

		subtree = type(self)(V, left, right)

		return subtree



	def child(self, i):
		return [self[i][t] for t in ["left", "right"] if not(self[i][t] is None)]