import random
from GLOBAL_VARS import *


def clean_dfa(q_0, dfa, T):
	"""
	This code removes transitions (and nodes) that are not used.
	All those transitions are forced to become self-loops
	"""
	not_used = set([k for k in dfa])
	for tau,g in T:
		q = q_0
		for sigma in tau:
			not_used.discard((q,sigma))
			if (q,sigma) in dfa:
				q = dfa[(q,sigma)]
	# removing transitions that were not used
	for r in not_used:
		del dfa[r]

def _update_ok(g,g_pos,ok,is_q0_pos,q,t):
	if q == 0:
		guess = is_q0_pos
	else:
		guess = (q%2 == 0)

	if MULTILABEL:
		if g_pos in g and guess == 1:
			ok[t] += 1.0
		elif g_pos not in g and guess == 0:
			ok[t] += 1.0
	else:
		if g == g_pos and guess == 1:
			ok[t] += 1.0
		elif g != g_pos and guess == 0:
			ok[t] += 1.0

def test_binary_accuracy(q_0, dfa, T, g_pos, is_q0_pos, always_returns_no, smoothing=True):
	"""
	Returns binary classification statistics for a given DFA:
		accuracy, precision, recall, true negative rate, false negative rate, f1 score
	"""
	correct = 0
	total = 0

	tp = 0 
	tn = 0
	fn = 0
	total_true_positive = 0
	total_positive_guess = 0
	fp = 0

	for tau,g in T:
		q = q_0 
		for sigma in tau:
			if (q, sigma) in dfa:
				q = dfa[(q, sigma)]

		if q == q_0:
			guess = is_q0_pos
		else:
			guess = (q % 2 == 0)


		if MULTILABEL:
			if g_pos in g and guess == 1:
				correct += 1
				tp += 1
			elif g_pos not in g and guess == 0:
				correct += 1

			total += 1

			if g_pos in g: 
				total_true_positive += 1

			if g_pos not in g and guess == 1:
				fp += 1

			if g_pos not in g and guess == 0:
				tn += 1

			if g_pos in g and guess == 0:
				fn += 1

			if guess == 1:
				total_positive_guess += 1
		else:
			if g == g_pos and guess == 1:
				correct += 1
				tp += 1
			elif g != g_pos and guess == 0:
				correct += 1

			total += 1

			if g == g_pos: 
				total_true_positive += 1

			if g != g_pos and guess == 1:
				fp += 1

			if g != g_pos and guess == 0:
				tn += 1

			if g == g_pos and guess == 0:
				fn += 1

			if guess == 1:
				total_positive_guess += 1


	acc = float(correct)/total

	if smoothing:
		recall = max((float(tp)-1) / total_true_positive, 0)
		tnr = max((float(tn)-1) / (total - total_true_positive), 1/2/(total - total_true_positive))

		if total - total_positive_guess == 0:
			fnr = 0.05
		else:
			fnr = max(min(float(fn) / (total - total_positive_guess), 1 - 1/(total- total_positive_guess)), 1/(total-total_positive_guess))
		
		if total_true_positive <= 5 and recall < 0.2:
			recall = 0.2

		if total_positive_guess == 0 or total_positive_guess == 1:
			precision = 0.5
		else:
			precision = max(min(float(tp)/total_positive_guess, 1-1/total_positive_guess), 1/total_positive_guess)

		f1_score = 2 * (precision * recall) / (precision + recall)

	else:
		recall = float(tp) / total_true_positive
		tnr = float(tn) / (total - total_true_positive)
		fnr = float(fn) / (total - total_positive_guess) if total - total_positive_guess > 0 else 0.05
		precision = float(tp)/total_positive_guess if total_positive_guess > 0 else 0.5
		f1_score = 0

	

	return (acc, precision, recall, tnr, fnr, f1_score)

def test_binary_convergence(q_0, dfa, T, g_pos, is_q0_pos):
	"""
	This code tests the binary classificaion performance of the automata over the set of traces T
	It uses the convergence analysis (shows the performance over time while the traces are progressively shown)
	"""
	total = float(len(T))
	ok    = [0.0]*(max([len(tau) for tau,_ in T]) + 1)
	for tau,g in T:
		t = 0
		q = q_0
		for sigma in tau:
			_update_ok(g,g_pos,ok,is_q0_pos,q,t)
			if (q,sigma) in dfa:
				# Recall that non-defined transitions are treated as self-loops
				q = dfa[(q,sigma)]

			t += 1
		# completing the rest of the trace with the final classification
		while t < len(ok):
			_update_ok(g,g_pos,ok,is_q0_pos,q,t)
			t+=1

	# normalizing the accuracy
	test_performance = [value/total for value in ok]
	return test_performance

def _update_multigoal_ok_bayes(g,q,DFAs,ok,ok_prob, t, topk = 1):
	"""
	Uses statistics from a validation set to perform Bayesian inference over the set of possible goals.
	See supplementary material from the paper for the derivation of this technique. 
	"""

	probs = {}

	sum_probs = 0
	probs_sorted = []

	for _g in DFAs:
		if DFAs[_g].guess(q[_g]) == 1:
			probs[_g] = DFAs[_g].goal_priors[_g] * DFAs[_g].recall / (1-DFAs[_g].tnr)
		else:
			probs[_g] = DFAs[_g].goal_priors[_g] * (1 - DFAs[_g].recall) / DFAs[_g].tnr

		sum_probs += probs[_g]

		probs_sorted.append((probs[_g], _g))


	probs_sorted.sort()
	probs_sorted.reverse()

	ok_prob[t] += probs[g] / sum_probs

	# Check if g is within the top k goals
	for i in range(min(topk, len(probs_sorted))):
		if probs_sorted[i][1] == g:
			ok[t] += 1
			break

def _update_multilabel_ok(gs,q,DFAs,ok, t):
	"""
	Only used for the multi-label experiments. 
	""" 

	assert(MULTILABEL)
	
	probs = {}

	acc = 0

	for _g in DFAs:
		if DFAs[_g].guess(q[_g]) == 1 and _g in gs:
			acc += 1
		elif DFAs[_g].guess(q[_g]) == 0 and _g not in gs:
			acc += 1
			
	ok[t] += acc / len(DFAs)


def test_multilabel_convergence(DFAs, T):
	"""
	Only used for the multi-label experiments.
	"""
	assert(MULTILABEL)
	total = float(len(T))
	ok    = [0.0]*(max([len(tau) for tau,_ in T]) + 1)

	for tau,gs in T:
		t = 0
		q = dict([(_g,DFAs[_g].q_0) for _g in DFAs])
		for sigma in tau:
			_update_multilabel_ok(gs,q,DFAs,ok, t)

			for _g in DFAs:
				if (q[_g],sigma) in DFAs[_g].dfa:
					# Recall that non-defined transitions are treated as self-loops
					q[_g] = DFAs[_g].dfa[(q[_g],sigma)]
			t += 1
		# completing the rest of the trace with the final classification
		while t < len(ok):
			_update_multilabel_ok(gs,q,DFAs,ok, t)
			t+=1
	# normalizing the accuracy
	test_performance = [value/total for value in ok]
	return test_performance

def get_prediction(q, DFAs):
	"""
	For a set of DFAs (one for each goal), returns the predicted goal and corresponding probability, under
	the Bayesian inference method
	"""

	probs = {}

	sum_probs = 0
	probs_sorted = []

	for _g in DFAs:
		if DFAs[_g].guess(q[_g]) == 1:
			probs[_g] = DFAs[_g].goal_priors[_g] * DFAs[_g].recall / (1-DFAs[_g].tnr)
		else:
			probs[_g] = DFAs[_g].goal_priors[_g] * (1 - DFAs[_g].recall) / DFAs[_g].tnr

		sum_probs += probs[_g]

		probs_sorted.append((probs[_g], _g))


	probs_sorted.sort()
	probs_sorted.reverse()

	return probs_sorted[0][1], probs_sorted[0][0] / sum_probs

def test_multigoal_convergence(DFAs, T, topk = 1):
	"""
	This code tests the multigoal performance over the set of traces T.
	It uses the convergence analysis (shows the performance over time while the traces are progressively shown).
	This uses the Bayesian inference technique.
	"""
	total = float(len(T))
	ok    = [0.0]*(max([len(tau) for tau,_ in T]) + 1)
	ok_prob = [0.0]*(max([len(tau) for tau,_ in T]) + 1)
	for tau,g in T:
		t = 0
		q = dict([(_g,DFAs[_g].q_0) for _g in DFAs])
		for sigma in tau:
			_update_multigoal_ok_bayes(g,q,DFAs,ok,ok_prob, t, topk = topk)

			for _g in DFAs:
				if (q[_g],sigma) in DFAs[_g].dfa:
					# Recall that non-defined transitions are treated as self-loops
					q[_g] = DFAs[_g].dfa[(q[_g],sigma)]
			t += 1
		# completing the rest of the trace with the final classification
		while t < len(ok):
			_update_multigoal_ok_bayes(g,q,DFAs,ok,ok_prob, t, topk = topk)
			t+=1
	# normalizing the accuracy
	test_performance = [value/total for value in ok]
	test_performance_prob = [value/total for value in ok_prob]
	return test_performance, test_performance_prob

def test_multigoal_percent_convergence(DFAs, T, topk = 1):
	"""
	Similar to `test_multigoal_convergence` but returns accuracy given 20%, 40%, 60%, 80%, and 100% of the full trace lengths
	"""
	total = float(len(T))

	conv_accuracy = [0,0,0,0,0] # accuracy at 20%, 40%, 60%, 80%, 100% of traces

	for tau,g in T:
		t = 0
		q = dict([(_g,DFAs[_g].q_0) for _g in DFAs])
		ok = [0.0 for i in range(len(tau)+1)]
		ok_prob = [0.0 for i in range(len(tau)+1)]

		for sigma in tau:
			_update_multigoal_ok_bayes(g,q,DFAs,ok,ok_prob, t, topk = topk)

			for _g in DFAs:
				if (q[_g],sigma) in DFAs[_g].dfa:
					# Recall that non-defined transitions are treated as self-loops
					q[_g] = DFAs[_g].dfa[(q[_g],sigma)]
			t+=1

		_update_multigoal_ok_bayes(g,q,DFAs,ok,ok_prob, t, topk = topk)

		conv_accuracy[0] += ok[max(1,int(len(tau)*0.2))]
		conv_accuracy[1] += ok[max(1,int(len(tau)*0.4))]
		conv_accuracy[2] += ok[max(1,int(len(tau)*0.6))]
		conv_accuracy[3] += ok[max(1,int(len(tau)*0.8))]
		conv_accuracy[4] += ok[max(1,int(len(tau)*1))]

	conv_accuracy[0] /= len(T)
	conv_accuracy[1] /= len(T)
	conv_accuracy[2] /= len(T)
	conv_accuracy[3] /= len(T)
	conv_accuracy[4] /= len(T)

	return conv_accuracy


"""
The following functions are only used in the early prediction experiments
""" 

def utility(t):
	return max(1 - t/40, 0)

def test_early_utility(DFAs, T):
	score = 0 
	time_percent = 0

	for tau, g in T:
		q = dict([(_g,DFAs[_g].q_0) for _g in DFAs])
		t = 0

		best_expected_utility = -1
		prediction_goal = None
		prediction_t = None

		for sigma in tau:
			for _g in DFAs:
				if (q[_g],sigma) in DFAs[_g].dfa:
					# Recall that non-defined transitions are treated as self-loops
					q[_g] = DFAs[_g].dfa[(q[_g],sigma)]
			t += 1			
			
			prediction, conf = get_prediction(q, DFAs)
			expected_utility = utility(t) * conf

			if expected_utility > best_expected_utility:
				best_expected_utility = expected_utility
				prediction_goal = prediction
				prediction_t = t

		if prediction_goal == g:
			score += utility(prediction_t)
			time_percent += prediction_t / len(tau)

	print("Average score:", score / len(T))
	print("Average time of prediction:", time_percent / len(T))

	return score / len(T)


class DFA:
	def __init__(self, q_0, dfa, confidence, g_pos, is_q0_pos, goal_priors):
		self.q_0 = q_0 
		self.dfa   = dfa
		if confidence is not None:
			self.accuracy = confidence[0]
			self.precision = confidence[1]
			self.recall = confidence[2]
			self.tnr = confidence[3]
		self.g_pos = g_pos
		self.is_q0_pos = is_q0_pos
		self.goal_priors = goal_priors

	def guess(self,q):
		"""
		Helper function to return the binary classification decision of this DFA,
		given the state
		"""
		if q == 0:
			if self.is_q0_pos:
				guess = 1
			else:
				guess = 0
		else:
			guess = (q % 2 == 0)

		return guess

def get_most_common_performance(T):
	"""
	Accuracy given by guessing the most common goal in the data.
	"""
	goals = {}
	for _,g in T:
		if g not in goals:
			goals[g] = 0
		goals[g] +=1
	return max([goals[g] for g in goals])/float(len(T))

def show_dfa(q_0, dfa, pos_prob):
	"""
	Show the DFA in a user-readable form
	"""
	print("\n----------- DFA -----------")
	transitions = [(i,dfa[(i,sigma)],sigma) for i,sigma in dfa]
	transitions.sort()
	for i,j,sigma in transitions:
		print(f"delta({i},{sigma}) = {j}")
	print(f"q_0 = {q_0}")
	# print(f"pos_prob = {pos_prob}")

def report_dfa(q_0, dfa, g):
	"""
	Give a complete description of the DFA as a string
	"""
	output = ""
	output += "----------- DFA ----------- " + "Goal: " + g + " ----------- \n"
	transitions = [(i,dfa[(i,sigma)],sigma) for i,sigma in dfa]
	transitions.sort()
	for i,j,sigma in transitions:
		output += (f"delta({i},{sigma}) = {j}\n")
	output += f"q_0 = {q_0}\n"
	return output
