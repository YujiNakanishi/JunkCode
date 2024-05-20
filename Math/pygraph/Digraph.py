"""
/****************/
Digraph
/****************/
class and function for digraph is defined.
"""

import pygraph
from pygraph.Undigraph import undigraph

"""
/***************/
digraph
/***************/
Field  : nodes -> list of dictionary.

Method : 
<inherited from Base> __getitem__, __len__, list, copy, equal, greater, index
<inherited from undigraph> __init__, append, isConnect, 
<defined in this class> remove, connect, disconnect, Degree

Note   :
--- Introduction ---
This is for direction graph.

--- Node ---
node has two keys of "value" and "adjacent".
 "value" is as explained if Base class.
 "adjacent" is list of connected node index. For exsample, if node i directs node p, q and r, 
node[i]["adjacent"] = [p, q, r].
"""
class digraph(undigraph):

	"""
	process : remove node i
	input   : i -> int. node index
	"""
	def remove(self, i):
		#####pop node i
		self.nodes.pop(i)

		for idx in range(len(self)): #for all nodes
			#####disconnect
			self.disconnect(idx, i)

			#####reindex
			for idxa in range(len(self[idx]["adjacent"])): #for all adjacents
				if self[idx]["adjacent"][idxa] > i:
					self[idx]["adjacent"][idxa] -= 1


	"""
	process : connect node i -> j
	input   : i, j -> int. node index
	"""
	def connect(self, i, j):
		if self.isConnect(i, j) == False: #This method is performed only if there isn't i -> j edge.
			self[i]["adjacent"].append(j)

	"""
	process : disconnect node i -> j
	input   : i, j -> int. node index
	"""
	def disconnect(self, i, j):
		if self.isConnect(i, j): #This method is performed only if there is i -> j edge..
			self[i]["adjacent"].remove(j)

	"""
	process : return degree of element i
	output : (outdegree, indegree)
	"""
	def Degree(self, i):
		outdegree = len(self[i]["adjacent"])
		indegree = sum([self.isConnect(t, i) for t in range(len(self))])

		return (outdegree, indegree)