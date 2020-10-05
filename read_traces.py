from random import random, randint, shuffle
from GLOBAL_VARS import *

def read_data(path):
	"""
	Read datasets from `path`. 
	"""

	G      = set() # set of possible goals
	Sigma  = set() # vocabulary of the automaton
	T      = []    # scenarios (i.e., list of traces with their respective goal)

	# reading the traces
	num_traces = {}
	f = open(path)
	for l in f:
		if MULTILABEL:
			tau,gs = tuple(l.rstrip().split(";"))
			# adding the goal to the set of available goals
			gs = gs.strip().split(",")

			for g in gs:
				G.add(g)
				if g not in num_traces:
					num_traces[g] = 0
				num_traces[g] += 1

			tau = tau.split(",")
			T.append((tau, gs))
			for sigma in tau:
				Sigma.add(sigma)		
		else:
			# reading the trace and its goal
			tau,g = tuple(l.rstrip().split(";"))
			# adding the goal to the set of available goals
			G.add(g)
			if g not in num_traces:
				num_traces[g] = 0
			num_traces[g] += 1

			tau = tau.split(",")
			T.append((tau,g))
			for sigma in tau:
				Sigma.add(sigma)
	f.close()

	return G, Sigma, T

def read_data_split(path):
	"""
	Split into training and validation sets. 
	20% of data for each goal goes into validation. 
	"""
	assert(not MULTILABEL)
	G      = set() # set of possible goals
	Sigma  = set() # vocabulary of the automaton
	T_train      = []    # scenarios (i.e., list of traces with their respective goal)
	T_validation = []
	# reading the traces
	traces = {}
	f = open(path)
	for l in f:
		# reading the trace and its goal
		tau,g = tuple(l.rstrip().split(";"))
		# adding the goal to the set of available goals
		G.add(g)
		if g not in traces:
			traces[g] = []

		tau = tau.split(",")
		traces[g].append(tau)

		for sigma in tau:
			Sigma.add(sigma)
	f.close()

	for g in G:
		n_valid = max(1, int(len(traces[g]) * 0.2))

		for i in range(n_valid):
			T_validation.append((traces[g][i], g))
		for i in range(n_valid, len(traces[g])):
			T_train.append((traces[g][i], g))

	return G, Sigma, T_train, T_validation

def count_observations(train_path, test_path):
	f_train, f_test = open(train_path), open(test_path)

	count = 0 
	total = 0
	for l in f_train:
		# reading the trace and its goal
		tau,g = tuple(l.rstrip().split(";"))
		count += len(tau.split(","))
		total += 1

	for l in f_test:
		# reading the trace and its goal
		tau,g = tuple(l.rstrip().split(";"))
		count += len(tau.split(","))
		total += 1

	return count, count / total, total

