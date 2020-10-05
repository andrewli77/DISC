"""
This file contains helper code for constructing the prefix tree of observations from training data.
"""

import read_traces, DFA_utils_tree_minerror, math, time
from random import choices, random
from GLOBAL_VARS import *

def get_reachable_nodes(node):
	"""
	returns a list with all the nodes from the tree with root *node*
	"""
	ret = []
	stack = [node]

	while len(stack) > 0:
		cur = stack.pop()
		ret.append(cur)
		for c in cur.get_children():
			stack.append(c)
	return ret

class TreeNode:
	def __init__(self, parent, p_sigma):
		self.parent   = parent  # reference to the parent node
		self.p_sigma  = p_sigma # reference to the sigma that moves the parent to this child
		self.children = {}
		self.positive = 0
		self.negative = 0
		self.trace_ends_here = 0

	def get_psigma(self):
		return self.p_sigma

	def add_label(self, is_positive, length):
		"""
		Here we normalize by the length of the full trace (so each trace has equal contribution to the error)
		"""
		if is_positive:
			self.positive += 1 / length
		else:
			self.negative += 1 / length

	def get_num_positive(self):
		return self.positive

	def get_num_negative(self):
		return self.negative

	def assign_id(self, n_id):
		self.n_id = n_id

	def get_id(self):
		return self.n_id

	def add_trace_end(self):
		self.trace_ends_here += 1

	def is_root(self):
		return self.parent is None

	def get_parent(self):
		return self.parent, self.p_sigma

	def get_children(self):
		return self.children.values()

	def get_child(self, sigma):
		if sigma not in self.children:
			# adding a new child
			self.children[sigma] = TreeNode(self, sigma)
		return self.children[sigma]

	def is_terminal(self):
		return self.positive == 0 or self.negative == 0

	def is_positive_node(self):
		return self.negative == 0

	def is_negative_node(self):
		return self.positive == 0

	def add_MIP_variables(self, MIP_vars):
		self.MIP_vars = MIP_vars

	def get_MIP_variables(self):
		return self.MIP_vars

	def must_be_terminal(self):
		return self.trace_ends_here > 0

def prune_tree(node):
	"""
	Reduces the tree into a prefix tree by terminating at all nodes
	where all traces are positive or all traces are negative.
	"""
	if node.is_terminal():
		node.children = {}
		return
	else:
		for c in node.get_children():
			prune_tree(c)

def create_tree(g_pos, G, Sigma, T, prune=True):
	"""
	Create the tree for label g_pos. Specify prune=True if you want to reduce it to
	a prefix tree.
	"""

	assert g_pos in G, f"Error, g_pos not in G"

	root = TreeNode(None, None)
	for tau,g in T:
		node = root
		if MULTILABEL:
			label_pos = (g_pos in g)
		else:
			label_pos = (g == g_pos)
		node.add_label(label_pos, len(tau))
		for sigma in tau:
			node = node.get_child(sigma)
			node.add_label(label_pos, len(tau))
		node.add_trace_end() # we mark that some trace ends in this node

	if prune:
		prune_tree(root)

		# Pad on the last observation on traces from where the prefix is cut off 
		for tau,g in T:
			if MULTILABEL:
				label_pos = (g_pos in g)
			else:
				label_pos = (g == g_pos)

			# For negative labels, we skip with high probability 
			if not label_pos:
				if random() < 0.95:
					continue

			node = root 
			for i in range(len(tau)):
				if len(node.get_children()) == 0:
					suffix = list(tau[i+1:-1])

					if len(suffix) > 0:
						node = node.get_child(tau[-1])
						node.add_label(label_pos, float(len(tau)) / len(suffix))
					break

				else:
					node = node.get_child(tau[i])
	return root
