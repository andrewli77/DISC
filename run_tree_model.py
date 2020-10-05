"""
This is the main file to run our approach, DISC, and the full-tree DFA learning method called DFA-FT in our paper. 
"""

import read_traces, DFA_utils_tree_minerror, DFA_utils_tree_only, time
import tree_only, tree_minerror
import argparse
from GLOBAL_VARS import *

def run_multigoal_tree_only(base, iteration, timeout=180):
	"""
	This code learns an automaton for every goal and test its performance. 
	It reports the convergence performance in the training and testing set.
	It also reports the performance of the always-yes and always-no baselines.

    Parameters
    -------
    base: string 
    	The base file prefix of the training and test sets. (e.g. "alfred")
    iteration: string
    	The unique identifier of the iteration for the train/test dataset (e.g. "1", "2", ...)
    timeout: int
		Time limit (in minutes) given to the model to learn the automaton
	"""
	path_trainset = "traces/" + base + "/train_" + iteration + ".txt" # Path to the file with the training traces
	path_testset = "traces/" + base + "/test_" + iteration + ".txt"

	# learning one DFA per goal
	if VALIDATION:
		G, Sigma, T_train, T_validation = read_traces.read_data_split(path_trainset)
	else:
		G, Sigma, T_train = read_traces.read_data(path_trainset)
		T_validation = []
	_, _, T_test = read_traces.read_data(path_testset)
	retrain_model = False

	avg_num_states = 0 
	avg_num_transitions = 0
	DFAs = {}
	t_init   = time.time()
	goal_priors = DFA_utils_tree_only.goal_priors(T_train + T_validation)

	for g_pos in G: 
		print("Working on goal", g_pos)

		q_0, dfa, pos_prob = tree_only.solve_tree_only(g_pos, G, Sigma, T_train, timeout, {})
		
		if VALIDATION:
			confidences = DFA_utils_tree_only.test_binary_accuracy(q_0, dfa, pos_prob, T_validation, g_pos)
		else:
			confidences = DFA_utils_tree_only.test_binary_accuracy(q_0, dfa, pos_prob, T_test, g_pos, smoothing=False)
			print("Confidence:", confidences)

		print("Goal:", g_pos)

		# Saving the automata
		DFAs[g_pos] = DFA_utils_tree_only.DFA(q_0, dfa, confidences, pos_prob, g_pos, goal_priors)
		sz = DFA_utils_tree_only.dfa_size(q_0, dfa, pos_prob)
		avg_num_states += sz[0]
		avg_num_transitions += sz[1]

	t_end = time.time()
	

	# Checking the train and test performance
	# Both values might be lower than 100%
	print("----------------------------------------")
	if VALIDATION:
		conv_train = DFA_utils_tree_only.test_multigoal_convergence_bayes(DFAs, T_train + T_validation)[0]
		conv_test = DFA_utils_tree_only.test_multigoal_convergence_bayes(DFAs, T_test)[0]
		conv_percent_test = DFA_utils_tree_only.test_multigoal_percent_convergence_bayes(DFAs, T_test)
		avg_utility = 0
	else:
		conv_train = DFA_utils_tree_only.test_multigoal_convergence(DFAs, T_train)
		conv_test = DFA_utils_tree_only.test_multigoal_convergence(DFAs, T_test)
		conv_percent_test = DFA_utils_tree_only.test_multigoal_percent_convergence(DFAs, T_test)
		avg_utility = 0
	conv_train = ["%0.3f"%v for v in conv_train]
	print(f"Train convergence performance: {conv_train}")

	print("----------------------------------------")

	conv_test = ["%0.3f"%v for v in conv_test]
	print(f"Test convergence performance: {conv_test}")	

	conv_percent_test = ["%0.3f"%v for v in conv_percent_test]
	print(f"Test convergence at 20%: {conv_percent_test[0]}")	
	print(f"Test convergence at 40%: {conv_percent_test[1]}")	
	print(f"Test convergence at 60%: {conv_percent_test[2]}")	
	print(f"Test convergence at 80%: {conv_percent_test[3]}")	
	print(f"Test convergence at 100%: {conv_percent_test[4]}")	

	most_common = DFA_utils_tree_only.get_most_common_performance(T_test)
	len([1 for tau,g in T_test if g==g_pos])/float(len(T_test))
	print(f"Most-Common-Class baseline: {most_common:0.3f}")
	print(f"Random classifier baseline: {1/float(len(G)):0.3f}")
	print("----------------------------------------")
	print(f"Time: {(t_end-t_init)/60:0.2f}[m]")
	print("Average num states:", avg_num_states / len(DFAs))
	print("Average num edges:", avg_num_transitions / len(DFAs))


	outf = open("traces/" + base + "/results/" + "tree_only_" + iteration + ".txt", "w")
	outf.write(", ".join(conv_test) + "; " + ", ".join(conv_percent_test) + "; " + str(avg_utility))
	outf.close()

	sz_f = open("traces/" + base + "/results/" + "tree_only_" + iteration + "_sizes.txt", "w")
	sz_f.write("%0.3f, %0.3f\n" % (avg_num_states / len(DFAs), avg_num_transitions / len(DFAs)))
	sz_f.close()


def run_multigoal_tree_minerror(base, iteration, q_max, timeout=180):
	"""
	This code learns an automaton for every goal and test its performance. 
	It reports the convergence performance in the training and testing set.
	It also reports the performance of the always-yes and always-no baselines.

    Parameters
    -------
    base: string 
    	The base file prefix of the training and test sets. (e.g. "alfred")
    iteration: string
    	The unique identifier of the iteration for the train/test dataset (e.g. "1", "2", ...)
    q_max: int
    	The maximum size for any automaton. 
    timeout: int
		Time limit (in minutes) given to the model to learn the automaton
	"""

	path_trainset = "traces/" + base + "/train_" + iteration + ".txt" # Path to the file with the training traces
	path_testset = "traces/" + base + "/test_" + iteration + ".txt"

	if VALIDATION:
		G, Sigma, T_train, T_validation = read_traces.read_data_split(path_trainset)
	else:
		G, Sigma, T_train = read_traces.read_data(path_trainset)
		T_validation = []

	_, _, T_test = read_traces.read_data(path_testset)

	DFAs = {}
	t_init   = time.time()
	for g_pos in G: 
		print("Working on goal", g_pos)
		best_hyperparams = None
		best_fscore = -1
		best_stats = None
		best_model = None

		for loop_penalty in [0.001]:
			for trans_penalty in [10, 3, 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]:
				q_0, dfa, is_q0_pos, always_returns_no, goal_priors = tree_minerror.solve_tree_minerror(g_pos, q_max, G, Sigma, T_train, timeout, {}, loop_penalty = loop_penalty, trans_penalty = trans_penalty)
				if VALIDATION:
					valid_stats = DFA_utils_tree_minerror.test_binary_accuracy(q_0, dfa, T_validation, g_pos, is_q0_pos, always_returns_no)
				else:
					valid_stats = DFA_utils_tree_minerror.test_binary_accuracy(q_0, dfa, T_train, g_pos, is_q0_pos, always_returns_no)

				if valid_stats[5] > best_fscore:
					best_fscore = valid_stats[5]
					best_hyperparams = (loop_penalty, trans_penalty)
					#best_conv = conv
					best_stats = valid_stats
					best_model = (q_0, dfa, is_q0_pos, always_returns_no, goal_priors)

		print(best_hyperparams)

		q_0, dfa, is_q0_pos, always_returns_no, goal_priors = best_model
		confidence = best_stats

		# Check if the DFA always returns no for any possible trace -- if so, modify the confidences accordingly. 
		if always_returns_no:
			confidence = (confidence[0], confidence[1], 0, 1, confidence[4], 0)

		test_acc = DFA_utils_tree_minerror.test_binary_convergence(q_0, dfa, T_test, g_pos, is_q0_pos)
		print("Goal:", g_pos, "Prior prob: ", goal_priors[g_pos])
		print("Accuracy/Precision/Recall/TNR/FNR/F2-score:", confidence[0], confidence[1],confidence[2], confidence[3], confidence[4], confidence[5])
		print("Real test accuracy:", test_acc)
		test_conf = DFA_utils_tree_minerror.test_binary_accuracy(q_0, dfa, T_test, g_pos, is_q0_pos, always_returns_no, smoothing = False)

		if always_returns_no:
			test_conf = (test_conf[0], test_conf[1], 0, 1)
		print("Real test accuracy/precision/recall/TNR", test_conf[0], test_conf[1], test_conf[2], test_conf[3])

		# Saving the automata
		DFA_utils_tree_minerror.show_dfa(q_0, dfa, is_q0_pos) 
		DFAs[g_pos] = DFA_utils_tree_minerror.DFA(q_0, dfa, confidence, g_pos, is_q0_pos, goal_priors)
	t_end = time.time()
	

	# Checking the train and test performance
	# Both values might be lower than 100%
	print("----------------------------------------")

	if VALIDATION:
		conv_train, conv_train_prob = DFA_utils_tree_minerror.test_multigoal_convergence(DFAs, T_train + T_validation)
	else:
		conv_train, conv_train_prob = DFA_utils_tree_minerror.test_multigoal_convergence(DFAs, T_train)
	conv_train = ["%0.3f"%v for v in conv_train]
	conv_train_prob = ["%0.3f"%v for v in conv_train_prob]
	print(f"Train convergence argmax performance: {conv_train}")
	print(f"Train convergence probability performance: {conv_train_prob}")

	print("----------------------------------------")
	conv_test, conv_test_prob = DFA_utils_tree_minerror.test_multigoal_convergence(DFAs, T_test, topk=1)
	conv_test = ["%0.3f"%v for v in conv_test]
	conv_test_prob = ["%0.3f"%v for v in conv_test_prob]
	print(f"Test convergence argmax performance: {conv_test}")	
	print(f"Test convergence probability performance: {conv_test_prob}")	

	print("----------------------------------------")
	conv_percent_test = DFA_utils_tree_minerror.test_multigoal_percent_convergence(DFAs, T_test, topk=1)
	conv_percent_test = ["%0.3f"%v for v in conv_percent_test]
	print(f"Test convergence at 20%: {conv_percent_test[0]}")	
	print(f"Test convergence at 40%: {conv_percent_test[1]}")	
	print(f"Test convergence at 60%: {conv_percent_test[2]}")	
	print(f"Test convergence at 80%: {conv_percent_test[3]}")	
	print(f"Test convergence at 100%: {conv_percent_test[4]}")	

	print("----------------------------------------")
	avg_utility = DFA_utils_tree_minerror.test_early_utility(DFAs, T_test)

	print("----------------------------------------")
	most_common = DFA_utils_tree_minerror.get_most_common_performance(T_test)
	len([1 for tau,g in T_test if g==g_pos])/float(len(T_test))
	print(f"Most-Common-Class baseline: {most_common:0.3f}")
	print(f"Random classifier baseline: {1/float(len(G)):0.3f}")

	print("----------------------------------------")
	print(f"Time: {(t_end-t_init)/60:0.2f}[m]")

	outf = open("traces/" + base + "/results/" + "ours_" + iteration + ".txt", "w")
	outf.write(", ".join(conv_test) + "; " + ", ".join(conv_percent_test) + "; " + str(avg_utility))
	outf.close()

	modelf = open("traces/" + base + "/results/" + "ours_" + iteration + "_models.txt", "w")
	for g in DFAs:
		modelf.write(DFA_utils_tree_minerror.report_dfa(DFAs[g].q_0, DFAs[g].dfa, g))
	modelf.close()


def run_multilabel_tree_minerror(base, iteration, q_max, timeout=180):
	"""
	This code is the version of run_tree_minerror for the multilabel experiment.
	"""
	assert(MULTILABEL)
	path_trainset = "traces/" + base + "/train_" + iteration + ".txt" # Path to the file with the training traces
	path_testset = "traces/" + base + "/test_" + iteration + ".txt"

	G, Sigma, T_train = read_traces.read_data(path_trainset)
	_, _, T_test = read_traces.read_data(path_testset)

	DFAs = {}
	for g_pos in G: 
		q_0, dfa, is_q0_pos, always_returns_no, goal_priors = tree_minerror.solve_tree_minerror(g_pos, q_max, G, Sigma, T_train, timeout, {}, loop_penalty = 0.001, trans_penalty = 0.0003)

		# Saving the automata
		DFA_utils_tree_minerror.show_dfa(q_0, dfa, is_q0_pos) 
		DFAs[g_pos] = DFA_utils_tree_minerror.DFA(q_0, dfa, None, g_pos, is_q0_pos, goal_priors)
	

	# Checking the train and test performance
	# Both values might be lower than 100%
	print("----------------------------------------")

	conv_train = DFA_utils_tree_minerror.test_multilabel_convergence(DFAs, T_train)
	conv_train = ["%0.3f"%v for v in conv_train]
	print(f"Train convergence performance: {conv_train}")

	print("----------------------------------------")

	conv_test = DFA_utils_tree_minerror.test_multilabel_convergence(DFAs, T_test)
	conv_test = ["%0.3f"%v for v in conv_test]
	print(f"test convergence performance: {conv_test}")

	outf = open("traces/" + base + "/results/" + "ours_" + iteration + ".txt", "w")
	outf.write(", ".join(conv_test))
	outf.close()


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', help='e.g. alfred, crystal')
	parser.add_argument('--qmax', type=int)
	parser.add_argument('--id', type=int)
	args = parser.parse_args()

	base = args.dataset
	q_max = args.qmax
	iteration = args.id

	# Default Parameters
	if base is None:
		base = "alfred"
	if q_max is None:
		q_max = 10
	if iteration is None:
		iteration = 1

	iteration = str(iteration)

	print("traces/" + base + "/train_" + iteration + ".txt")
	if TREE_MINERROR:
		if MULTILABEL:
			run_multilabel_tree_minerror(base, iteration, q_max, timeout=15)
		else:
			run_multigoal_tree_minerror(base, iteration, q_max, timeout=15)
	else:
		run_multigoal_tree_only(base, iteration, timeout=15)

	print("traces/" + base + "/train_" + iteration + ".txt")

## mit: (0.01, 0.5)

