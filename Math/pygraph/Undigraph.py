"""
/****************/
Undigraph
/****************/
class and function for undigraph is defined.
"""

import pygraph

"""
/***************/
undigraph
/***************/
Field  : nodes -> list of dictionary.

Method : 
<inherited from Base> __getitem__, __len__, list, copy, equal, greater, index
<defined in this class> append, remove, isConnect, connect, disconnect, Degree

Note   :
--- Introduction ---
This is for undirection graph.

--- Node ---
node has two keys of "value" and "adjacent".
 "value" is as explained if Base class.
 "adjacent" is list of connected node index. For exsample, if node i connects node p, q and r, 
node[i]["adjacent"] = [p, q, r] and i is in node[p~r]["adjacent"].
"""
class undigraph(pygraph.Base):

	"""
	input : V, E, nodes
		V -> value of list.
		E -> adjacent information. List of edge. elements are also list.
		     For exsample, if E[i] = [p, q] -> nodes[p] and nodes[q] are connected.
		nodes -> list of dictionary.
	Note :
	nodes != None in only case that this is called from method copy. 
	Therefore, if you define data structure explicity in your main code, nodes should be None.
	 If nodes = None, it should be V != None.
	"""
	def __init__(self, V = None, E = None, nodes = None):
		if nodes is None:
			#####define nodes and set value first.
			self.nodes = [{"value" : v, "adjacent" : []} for idx, v in enumerate(V)]

			#####define adjacent if E != None
			if not(E is None):
				for e in E:
					self.connect(e[0], e[1])

		else:
			##### copy method is called.
			self.nodes = nodes

	"""
	process : append a new node
	input   : v, E
		v -> any type. node value
		E -> list of connected node index
			if E is None, isolated node
	"""
	def append(self, v, E = None):
		self.nodes.append({"value" : v, "adjacent" : []})

		if not(E is None):
			for e in E:
				self.connect(e[0], e[1])

	"""
	process : remove node i
	input   : i -> int. node index
	"""
	def remove(self, i):
		##### make node i isolated
		for adj in self[i]["adjacent"]:
			self.disconnect(i, adj)

		#####pop node i
		self.nodes.pop(i)

		#####reindex
		for idx in range(len(self)): #for all nodes
			for idxa in range(len(self[idx]["adjacent"])): #for all adjacent
				if self[idx]["adjacent"][idxa] > i:
					self[idx]["adjacent"][idxa] -= 1


	"""
	process : check whether node i and j is connected or not.
	input   : i, j -> int. node index
	output  : boolean  
	"""
	def isConnect(self, i, j):
		return (j in self[i]["adjacent"])

	"""
	process : connect node i and j
	input   : i, j -> int. node index
	"""
	def connect(self, i, j):
		if self.isConnect(i, j) == False: #This method is performed only if node i and j isn't connected together.
			self[i]["adjacent"].append(j)
			self[j]["adjacent"].append(i)

	"""
	process : disconnect node i and j
	input   : i, j -> int. node index
	"""
	def disconnect(self, i, j):
		if self.isConnect(i, j): #This method is performed only if node i and j is connected together.
			self[i]["adjacent"].remove(j)
			self[j]["adjacent"].remove(i)

	"""
	process : return degree of node i.
	input   : i -> int. node index
	output  : int. degree  
	"""
	def Degree(self, i):
		return len(self[i]["adjacent"])
