"""
/****************/
RootedTree
/****************/
class and function for undigraph is defined.
"""

import sys
import pygraph
import random

"""
/***************/
undigraph
/***************/
Field  : nodes -> list of dictionary.

Method : 
<inherited from Base> __getitem__, __len__, list, copy, equal, greater, index 
<defined in this class> __init__, reindex, isConnect, connect, isAncestor, isDescendant, Root, child, Degree,
                        Sibling, Height, Depth, Leaf, Ancestors, Descendants, append, remove, Subtree, isLeaf,
                        isRoot

Note   :
--- Introduction ---
This is for direction graph.

--- Node ---
node has three keys of "value", "parent" and "child".
 "value" is as explained if Base class.
 "parent" is int or None. parent node index. if None, this node is root.
 "child" is list of int. child node indice. if void, this node is leaf.
"""
class rootedtree(pygraph.Base):
	"""
	input : V, parent, nodes
		V -> value of list.
		parent -> parent node indice. List of int.
		nodes -> list of dictionary.
	"""
	def __init__(self, V = None, parent = None, nodes = None):
		if nodes is None:
			self.nodes = [{"value" : v, "parent" : p, "child" : []} for v, p in zip(V, parent)]

			for idx in range(len(self)):
					if self.isRoot(idx) == False:
						p = self[idx]["parent"]
						self[p]["child"].append(idx)

		else:
			##### copy method is called.
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
			if self.isRoot(idx) == False:
				nodes[idx]["parent"] = node_index.index(nodes[idx]["parent"])
			
			for cidx in range(len(nodes[idx]["child"])):
				nodes[idx]["child"][cidx] = node_index.index(nodes[idx]["child"][cidx])

		self.nodes = nodes

	"""
	Process : append new node
	input : v, parent, child
		v -> any type of value
		parent, child -> int
	"""
	def append(self, v, parent, child):
		new_index = len(self)
		self.nodes.append({"value" : v, "parent" : parent, "child" : [child]})


		if not(parent is None):
			self[parent]["child"].append(new_index)
			self[parent]["child"].remove(child)

		self[child]["parent"] = new_index

	"""
	Process : remove node i
	"""
	def remove(self, i):
		children = self.child(i)
		parent = self[i]["parent"]
		self[parent]["child"].remove(i)
		self[parent]["child"] += children

		self.nodes.pop(i)

		for c in children:
			self(c)["parent"] = parent

		for idx in range(len(self)):
			if self.isRoot(idx) == False:
				if self[idx]["parent"] > i:
					self[idx]["parent"] -= 1

			for idxa in range(len(self[idx]["child"])):
				if self[idx]["child"][idxa] > i:
					self[idx]["child"][idxa] -= 1

	"""
	process : check whether nodes i and nodej are connected or not
	input   : i, j -> int
	output  : boolean
	"""
	def isConnect(self, i, j):
		adjacent = self.child(i)

		parent = self[i]["parent"]
		if not(parent is None):
			adjacent.append(parent)

		return (j in adjacent)

	"""
	process : check whether nodes i is leaf or not
	input   : i -> int
	output  : boolean
	"""
	def isLeaf(self, i):
		return (self.child(i) == [])

	"""
	process : check whether nodes i is root or not
	input   : i -> int
	output  : boolean
	"""
	def isRoot(self, i):
		return (self[i]["parent"] is None)

	"""
	process : return whether ancestor or not
	Input : i, j ->  int
	return whether node[j] is ancestor for node[i] or not
	"""
	def isAncestor(self, i, j):
		flag = False
		while self.isRoot(idx) == False:
			parent = self[i]["parent"]
			if parent == j:
				flag = True
				break
			else:
				i = parent

		return flag

	"""
	process : return whether node[j] is descendant for node[i] or not
	Input : i, j -> int
	"""
	def isDescendant(self, i, j):
		flag = [False]
		def _recisDescendant(p, q, flag):
			if flag[0] == False:
				child = self.child(p)
				if q in child:
					flag[0] = True
				else:
					for c in child:
						_recisDescendant(c, q, flag)
		_recisDescendant(i, j, flag)

		return flag[0]

	"""
	/*************/
	Root
	/*************/
	process : return index of root
	"""
	def Root(self, idx = None):
		if idx is None:
			idx = random.randint(0, len(self)-1)
		
		while self.isRoot(idx) == False:
			idx = self[idx]["parent"]

		return idx

	"""
	/****************/
	Ancestors
	/****************/
	process : return ancestors of nodes[i]
	"""
	def Ancestors(self, i):
		ancestor = []
		while self.isRoot(i) == False:
			ancestor.append(self[i]["parent"])
			i = ancestor[-1]

		return ancestor


	"""
	process : return descendants of nodes[i]
	"""
	def Descendants(self, i):
		descendant = []
		def _recDescendants(j):
			child = self.child(j)
			for c in child:
				_recDescendants(c)
			descendant.append(j)

		_recDescendants(i)
		descendant.remove(i)

		return descendant

	"""
	process : return depth of ith node
	"""
	def Depth(self, i):
		return len(self.Ancestors(i))

	"""
	process : return height of node i
	"""
	def Height(self, i):
		if self.isLeaf(i):
			return 0
		else:
			return max([self.Height(c)+1 for c in self.child(i)])


	"""
	process : return indice of children
	input   : i -> int
	output  : list of index
	"""
	def child(self, i):
		return self[i]["child"]

	"""
	process : return sibling of node i
	input   : i -> int
	output  : list of index
	"""
	def Sibling(self, i):
		parent = self[i]["parent"]
		if parent is None:
			return []
		else:
			return self.child(parent)

	"""
	process : return index of leaf
	"""
	def Leaf(self):
		return [idx for idx in range(len(self)) if self.isLeaf(idx)]

	"""
	process : return degree of nodes[i]
	"""
	def Degree(self, i):
		return len(self.child(i))

	"""
	process : return subtree of node root_index
	"""
	def Subtree(self, root_index):
		index = self.Descendants(root_index) + [root_index]
		V = [self[idx]["value"] for idx in index]

		parents = [index.index(self[idx]["parent"]) for idx in index[:-1]] + [None]
		subtree = type(self)(V, parents)

		return subtree