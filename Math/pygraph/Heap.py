"""
/****************/
BinaryTree
/****************/
class and function for binarytree is defined.
"""

import pygraph

"""
process : create heap from binarytree
input : bitree, deg
	bitree -> binarytree class
	deg -> str. "max" or "min"
"""
def createHeap(bitree, deg = "max"):
	bitree_cpy = bitree.copy()
	
	if deg == "max":
		heap = maxheap(nodes = bitree_cpy.nodes)
	else:
		heap = minheap(nodes = bitree_cpy.nodes)

	heap.build()
	heap.reindex()

	return heap


"""
/***************/
maxheap
/***************/
Field  : nodes -> list of dictionary.

Method : 
<inherited from Base> __getitem__, __len__, list, copy, equal, greater, index 
<inherited from rootedtree> isConnect, Root, Degree, Leaf, Sibling, Height, Depth, isAncestor, isDescendant, 
                            Descendants, Ancestors
<defined from binarytree> __init__, child, reindex, append, remove, Subtree
<defined in this class> heapify, build
Note   :
--- Introduction ---
This is for  maxheap.

--- Node ---
node has four keys of "value", "parent", "left" and "right".
 "value" is as explained if Base class.
 "parent" is int or None. parent node index. if None, this node is root.
 "left" and "right" is int or None. left(right) node index.
"""
class maxheap(pygraph.BinaryTree.binarytree):
	"""
	process : heapify toward index i
	"""
	def heapify(self, i):
		left_index = self[i]["left"]
		right_index = self[i]["right"]
		if not(left_index is None):
			if (self[left_index]["value"] > self([i])["value"]):
				largest = left_index
			else:
				largest = i
		else:
			largest = i

		if not(right_index is None):
			if (self[right_index]["value"] > self[largest]["value"]):
				largest = right_index

		if largest != i:
			self[i]["value"], self[largest]["value"] = self[largest]["value"], self[i]["value"]
			self.heapify(largest)

	"""
	process : heapify toward all
	"""
	def build(self):
		for i in range(len(self)-1, -1, -1):
			self.heapify(i)



class minheap(maxheap):

	def heapify(self, i):
		left_index = self[i]["left"]
		right_index = self[i]["right"]

		if not(left_index is None):
			if (self[left_index]["value"] < self[i]["value"]):
				smallest = left_index
			else:
				smallest = i
		else:
			smallest = i

		if not(right_index is None):
			if (self[right_index]["value"] < self[smallest]["value"]):
				smallest = right_index

		if smallest != i:
			self[i]["value"], self[smallest]["value"] = self[smallest]["value"], self[i]["value"]
			self.heapify(smallest)