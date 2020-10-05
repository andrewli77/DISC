"""
This model minimizes the training error given a maximum size for the DFA
"""

from gurobipy import *
import read_traces, DFA_utils_tree_minerror, time, tree_utils
from GLOBAL_VARS import *

def solve_tree_minerror(g_pos, q_max, G, Sigma, T, timeout, info = {}, be_quiet=True, loop_penalty = 0.01, trans_penalty = 0.1):
	"""
    Parameters
    -------
    g_pos: string
    	The goal for which we are training an automata to discriminate
    q_max: int
    	The maximum number of DFA states
    G: set(string)
    	The set of all possible goals
    Sigma: set(string)
    	The set of all possible observations
    T:	list(tuple(trace, goal))
    	List of training data, each is an observation trace and goal.
    loop_penalty: float
    	The penalty assigned to state occupancy in a non-absorbing (i.e. not state 1, or 2) state
    trans_penalty: float
    	The penalty assigned for each transition
	"""

	assert g_pos in G, f"Error, g_pos not in G"
	# creating the auxiliary tree structure
	tree = tree_utils.create_tree(g_pos, G, Sigma, T)
	nodes = tree_utils.get_reachable_nodes(tree)
	print("nodes:", len(nodes))
	print("num_positive:", tree.get_num_positive())
	print("num_negative:", tree.get_num_negative())

	num_pos = tree.get_num_positive()
	num_neg = tree.get_num_negative()
	N = num_pos + num_neg

	if q_max < 0: 
		# Setting q_max to its maximum possible value
		q_max = len(nodes)
	else:
		q_max = min(len(nodes), q_max)

	q_max = max(q_max, 3)
	assert q_max >= 3, f"At least 3 nodes are needed for a one-vs-all classification"
	print("q_max: ", q_max)

	# We label the q0 node with the value of the most common class 
	always_yes = len([1 for tau,g in T if g==g_pos])/float(len(T))
	is_q0_pos = always_yes > 0.5

	# initializing a MIP model and auxiliary variables
	t_init = time.time()
	m = Model("extensive_form")
	if be_quiet:
		m.Params.outputFlag = 0    # turn off output
	#m.Params.method    = 2      # barrier method
	#m.Params.Threads   = 1
	q_0     = 0
	Q_all   = range(q_max) # odd ids are negative and even ids are positive. 0 is a special case. 

	# delta-variables
	delta = {} # automaton transitions
	for i in Q_all:
		for sigma in Sigma:
			for j in Q_all:
				ub = 1
				delta[(i,sigma,j)] = m.addVar(ub=ub, vtype=GRB.BINARY)

	# automaton is deterministic
	for i in Q_all:
		for sigma in Sigma:
			m.addConstr(sum([delta[(i,sigma,j)] for j in Q_all]) == 1)
			if i == 1 or i == 2:
				m.addConstr(delta[(i, sigma, i)] == 1)
	
	# node-variables
	for node in nodes:
		if node.is_root():
			n_vars = [(1 if i == 0 else 0) for i in Q_all]
		else:
			n_vars = [m.addVar(vtype=GRB.BINARY) for i in Q_all]
			# Constraint, the automaton is at one state on every node
			m.addConstr(sum(n_vars) == 1)
		node.add_MIP_variables(n_vars)

	# adding constraints
	for node in nodes:
		if node.is_root():
			continue
		parent, p_sigma = node.get_parent()
		p_vars = parent.get_MIP_variables()
		n_vars = node.get_MIP_variables()
		for i in Q_all:
			for j in Q_all:
				m.addConstr(p_vars[i] + n_vars[j] - 1 <= delta[(i,p_sigma,j)])

	# setting the objective function to minimize the prediction error per step
	total_error = []
	p = num_pos / N 	
	new_p = max(p, 0.15) # We reweight traces by label so positive traces account for at least 15% of the error
						 # otherwise it often learns DFAs that always predict no. 

	for node in nodes:
		n_vars = node.get_MIP_variables()
		for j in Q_all:
			if MULTILABEL:
				if j == 0:
					if is_q0_pos:
						# positive class
						total_error.append((node.get_num_negative() + loop_penalty) * n_vars[j])
					else:
						# negative class
						total_error.append((node.get_num_positive() + loop_penalty) * n_vars[j])

				elif j == 1:
					total_error.append(node.get_num_positive() * n_vars[j])
				elif j == 2:
					total_error.append(node.get_num_negative() * n_vars[j])

				elif j%2==0:
					# positive class
					total_error.append((node.get_num_negative() + loop_penalty) * n_vars[j])
				elif j%2==1:
					# negative class
					total_error.append((node.get_num_positive() + loop_penalty) * n_vars[j])

			else:
				if j == 0:
					if is_q0_pos:
						# positive class
						total_error.append(((1 - new_p) * N / num_neg * node.get_num_negative() + loop_penalty) * n_vars[j])
					else:
						# negative class
						total_error.append((new_p * N / num_pos * node.get_num_positive() + loop_penalty) * n_vars[j])

				elif j == 1:
					total_error.append(new_p * N / num_pos * node.get_num_positive() * n_vars[j])
				elif j == 2:
					total_error.append((1 - new_p) * N / num_neg * node.get_num_negative() * n_vars[j])

				elif j%2==0:
					# positive class
					total_error.append(((1 - new_p) * N / num_neg * node.get_num_negative() + loop_penalty) * n_vars[j])
				elif j%2==1:
					# negative class
					total_error.append((new_p * N / num_pos * node.get_num_positive() + loop_penalty) * n_vars[j])

	# we also have to penalize non-self-loop transitions
	trans_penalizations = []
	for i in Q_all:
		for sigma in Sigma:
			for j in Q_all:
				if i == j:
					continue
				trans_penalizations.append(delta[(i,sigma,j)])
	m.setObjective(sum(total_error) + trans_penalty * sum(trans_penalizations), GRB.MINIMIZE)
	m.update()
	m.Params.TimeLimit = timeout*60 - (time.time() - t_init) # discounting the time used building the model
	m.optimize()
	
	t_end = time.time()



	# return the automata (q_0, q_pos, q_neg, dfa)
	# NOTE: self-loops transitions are not included (the default is to self-loop)
	dfa = {}
	for i in Q_all:
		for sigma in Sigma:
			for j in Q_all:
				if delta[(i,sigma,j)].x > 0.99 and i != j:
					dfa[(i,sigma)] = j
	DFA_utils_tree_minerror.clean_dfa(q_0, dfa, T)

	if MULTILABEL:
		always_returns_no = False
		goal_priors = None
	else:
		# Check if this DFA always returns No 
		reachable = [False for q in Q_all]
		reachable[q_0] = True

		always_returns_no = not is_q0_pos


		for n in range(len(Q_all)):
			if not always_returns_no:
				break

			for q in Q_all:
				if reachable[q]:
					for sigma in Sigma:
						if (q, sigma) in dfa:
							reachable[dfa[(q, sigma)]] = True
							if dfa[(q, sigma)] % 2 == 0:
								always_returns_no = False
								break

		# updating the info
		info["objective"]  = m.ObjVal
		info["bound"]      = m.ObjBound
		info["gap"]        = m.MIPGap
		info["is_optimal"] = (m.status == GRB.OPTIMAL)
		info["num_nodes"]  = m.NodeCount
		info["num_vars"]   = m.NumIntVars + m.NumBinVars
		info["time"]       = (t_end-t_init)/60
		if m.SolCount > 0:
			print("objective: %0.2f"%info["objective"])
			print("bound: %0.2f"%info["bound"])
			print("gap: %0.2f"%info["gap"])

		# The prior probability of each goal in the dataset.
		goal_priors = {}
		for _, g in T:
			if g not in goal_priors:
				goal_priors[g] = 1
			else:
				goal_priors[g] += 1

		for g in goal_priors:
			goal_priors[g] /= len(T)


	return q_0, dfa, is_q0_pos, always_returns_no, goal_priors