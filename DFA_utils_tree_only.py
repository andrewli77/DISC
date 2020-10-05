import random

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

def add_probabilities(q_0, dfa, T, g_pos):
	"""
	This code adds probabilities to each of the intermediate nodes.
	"""
	pos_prob = {}
	total    = {}
	for tau,g in T:
		q = q_0
		q_vis = set([q])
		if q not in pos_prob:
			pos_prob[q] = 0
		for sigma in tau:
			if (q,sigma) in dfa:
				q = dfa[(q,sigma)]
			q_vis.add(q)
			if q not in pos_prob:
				pos_prob[q] = 0
		# adding the visited states to the total count
		for q in q_vis:
			if g == g_pos:
				pos_prob[q] += 1
			if q not in total:
				total[q] = 0
			total[q] += 1
	# normalizing the probabilities
	for q in pos_prob:
		pos_prob[q] = float(pos_prob[q]) / total[q]
	return pos_prob

def test_dfa(q_0, q_pos, q_neg, dfa, T, g_pos):
	"""
	This code tests the performance of the automata over the set of traces T
	"""
	ok    = 0.0
	total = 0.0
	for tau,g in T:
		q = q_0
		for sigma in tau:
			wrong = True
			if (q,sigma) in dfa:
				# Recall that non-defined transitions are treated as self-loops
				q = dfa[(q,sigma)]
			if q in [q_pos,q_neg]:
				if (g == g_pos and q == q_pos) or (g != g_pos and q == q_neg):
					ok += 1.0
					wrong = False
				break
		if wrong:
			print(tau)
			exit(0)
		total += 1.0
	return ok*100/total

def _update_ok(g,g_pos,ok,pos_prob,q,t):
	"""
	guessing the class using the probabilities
	"""
	p_guess = 1.0 if pos_prob[q] > 0.5 else 0.0
	if g == g_pos: ok[t] += p_guess
	else:          ok[t] += 1.0 - p_guess

def test_binary_convergence(q_0, dfa, T, g_pos, pos_prob):
	"""
	This code tests the performance of the automata over the set of traces T
	It uses the convergence analysis (shows the performance over time while the traces are progressively shown)
	"""
	total = float(len(T))
	ok    = [0.0]*(max([len(tau) for tau,_ in T]) + 1)
	for tau,g in T:
		t = 0
		q = q_0
		for sigma in tau:
			_update_ok(g,g_pos,ok,pos_prob,q,t)
			if (q,sigma) in dfa:
				# Recall that non-defined transitions are treated as self-loops
				q = dfa[(q,sigma)]
			t += 1
		# completing the rest of the trace with the final classification
		while t < len(ok):
			_update_ok(g,g_pos,ok,pos_prob,q,t)
			t+=1

	# normalizing the accuracy
	test_performance = [value/total for value in ok]
	return test_performance


def _update_multigoal_ok(g,q,DFAs,ok,t):
	"""
	guessing the class using the probabilities
	"""
	max_prob = max([DFAs[_g].pos_prob[q[_g]] for _g in DFAs])
	g_guess = [_g for _g in DFAs if DFAs[_g].pos_prob[q[_g]]==max_prob][0]
	if g == g_guess: 
		ok[t] += 1.0

def _update_multigoal_ok_bayes(g,q,DFAs,ok,ok_prob, t, topk = 1):
	"""
	guessing the class using the Bayesian inference technique 
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

def test_multigoal_convergence(DFAs, T):
	"""
	This code tests the multigoal performance over the set of traces T
	It uses the convergence analysis (shows the performance over time while the traces are progressively shown)
	
    Parameters
    -------
    DFAs[g_pos] = q_0, q_pos, q_neg, dfa, pos_prob
	"""
	total = float(len(T))
	ok    = [0.0]*(max([len(tau) for tau,_ in T]) + 1)
	for tau,g in T:
		t = 0
		q = dict([(_g,DFAs[_g].q_0) for _g in DFAs])
		for sigma in tau:
			_update_multigoal_ok(g,q,DFAs,ok,t)
			for _g in DFAs:
				if (q[_g],sigma) in DFAs[_g].dfa:
					# Recall that non-defined transitions are treated as self-loops
					q[_g] = DFAs[_g].dfa[(q[_g],sigma)]
			t += 1
		# completing the rest of the trace with the final classification
		while t < len(ok):
			_update_multigoal_ok(g,q,DFAs,ok,t)
			t+=1

	# normalizing the accuracy
	test_performance = [value/total for value in ok]
	return test_performance

def test_multigoal_convergence_bayes(DFAs, T, topk = 1):
	"""
	This code tests the multigoal performance over the set of traces T
	It uses the convergence analysis (shows the performance over time while the traces are progressively shown)
	
    Parameters
    -------
    DFAs[g_pos] = q_0, q_pos, q_neg, dfa, pos_prob
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

		for sigma in tau:
			_update_multigoal_ok(g,q,DFAs,ok,t)
			for _g in DFAs:
				if (q[_g],sigma) in DFAs[_g].dfa:
					# Recall that non-defined transitions are treated as self-loops
					q[_g] = DFAs[_g].dfa[(q[_g],sigma)]
			t += 1

		_update_multigoal_ok(g,q,DFAs,ok,t)

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

def test_multigoal_percent_convergence_bayes(DFAs, T, topk = 1):
	"""
	Similar to `test_multigoal_convergence_bayes` but returns accuracy given 20%, 40%, 60%, 80%, and 100% of the full trace lengths
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
			t += 1

		_update_multigoal_ok_bayes(g,q,DFAs,ok,ok_prob, t, topk = topk)
		# completing the rest of the trace with the final classification

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

def goal_priors(T):
	"""
	Return the prior probability of each goal given no other information
	"""
	goal_priors = {}
	for _, g in T:
		if g not in goal_priors:
			goal_priors[g] = 1
		else:
			goal_priors[g] += 1

	for g in goal_priors:
		goal_priors[g] /= len(T)
	return goal_priors

def test_binary_accuracy(q_0, dfa, pos_prob, T, g_pos, smoothing=True):
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

		guess = pos_prob[q] >= 0.5

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
		fnr = float(fn) / (total - total_positive_guess) if total - total_positive_guess > 0 else -1
		precision = float(tp)/total_positive_guess if total_positive_guess > 0 else -1
		f1_score = 0

	

	return (acc, precision, recall, tnr, fnr, f2_score)

class DFA:
	def __init__(self, q_0, dfa, confidence, pos_prob, g_pos, goal_priors):
		self.q_0 = q_0 
		self.dfa   = dfa
		if confidence is not None:
			self.accuracy = confidence[0]
			self.precision = confidence[1]
			self.recall = confidence[2]
			self.tnr = confidence[3]
		self.pos_prob = pos_prob
		self.g_pos = g_pos
		self.goal_priors = goal_priors

	def guess(self,q):
		return self.pos_prob[q] >= 0.5


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

def dfa_size(q_0, dfa, pos_prob):
	return len(pos_prob), len(dfa)

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
	print(f"pos_prob = {pos_prob}")