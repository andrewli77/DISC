"""
This code returns a DFA that is equivalent to the Tree constructed by compressing all the traces into one tree.
"""

import read_traces, DFA_utils_tree_only, time, tree_utils


def solve_tree_only(g_pos, G, Sigma, T, timeout, info, be_quiet=False):
	assert g_pos in G, f"Error, g_pos not in G"

	# creating the auxiliary tree structure
	tree = tree_utils.create_tree(g_pos, G, Sigma, T, prune=False)
	nodes = tree_utils.get_reachable_nodes(tree)

	# creating an equivalent DFA
	q_0     = 0
	q_pos   = 1
	q_neg   = 2

	# assigning ids to each node
	n_current = 3
	for n in nodes:
		if n.is_root():
			n.assign_id(q_0)
		elif n.is_positive_node():
			n.assign_id(q_pos)
		elif n.is_negative_node():
			n.assign_id(q_neg)
		else:
			n.assign_id(n_current)
			n_current += 1

	# creating the dfa
	dfa = {}
	for ni in nodes:
		if ni.is_terminal():
			continue
		ni_id = ni.get_id()
		for nj in ni.get_children():
			nj_id = nj.get_id()
			ni_sigma = nj.get_psigma()
			dfa[(ni_id,ni_sigma)] = nj_id
	DFA_utils_tree_only.clean_dfa(q_0, dfa, T)

	# Adding the probabilities
	pos_prob = DFA_utils_tree_only.add_probabilities(q_0, dfa, T, g_pos)

	return q_0, dfa, pos_prob
