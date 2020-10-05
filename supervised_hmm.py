"""
This file uses pomegranate's implementation of HMM to perform supervised multi-goal classification.
We learn a separate HMM (unsupervised) for traces from each label, and classify by choosing the HMM with highest
probability. 
"""


from random import shuffle 
from pomegranate import *
import numpy as np 
from GLOBAL_VARS import *

class HMM():
	def __init__(self, base, iteration):
		self.base = base
		self.iteration = iteration

		train_file = "traces/" + base + "/train_" + str(iteration) + ".txt"
		test_file = "traces/" + base + "/test_" + str(iteration) + ".txt"
		self.train_traces_by_goal = {}

		self.validation_traces = []
		self.validation_labels = []

		self.test_traces = []
		self.test_labels = []
	
		self.count = 0
		self.parse_train_test(train_file, test_file)

		# We have one model for each possible goal/label
		self.models = {} 

	def parse_train_test(self, train_file, test_file):
		train_data = open(train_file).read().strip().split("\n")
		test_data = open(test_file).read().strip().split("\n")

		for line in train_data:
			t,g = line.split(";")
			t = t.strip().split(",")

			if g in self.train_traces_by_goal:
				self.train_traces_by_goal[g].append(np.array(t))
			else:
				self.train_traces_by_goal[g] = [np.array(t)]

		for goal in self.train_traces_by_goal:
			if VALIDATION:
				n_validation = max(int(len(self.train_traces_by_goal[goal]) * 0.2), 1)
			else:
				n_validation = 0
			for i in range(n_validation):
				self.validation_traces.append(self.train_traces_by_goal[goal][i])
				self.validation_labels.append(goal)

			self.train_traces_by_goal[goal] = self.train_traces_by_goal[goal][n_validation:]


		for line in test_data:
			t,g = line.split(";")
			t = t.strip().split(",")

			self.test_traces.append(np.array(t))
			self.test_labels.append(g)

	def train_hmms(self):
		best_validation_acc = -1

		for n_components in [5, 10]:
			for pseudocount in [0, 0.1, 1]:
				models = {}

				for g in self.train_traces_by_goal:
					print(g, len(self.train_traces_by_goal[g]))
					X = self.train_traces_by_goal[g]
					
					models[g] = HiddenMarkovModel.from_samples(distribution=DiscreteDistribution, n_components=n_components, X=X, algorithm='baum-welch', pseudocount=pseudocount, verbose=False, stop_threshold=1e-3, max_iterations=1e6)
					models[g].bake(merge="None")

				validation_acc = self.validation(models)
				
				if validation_acc > best_validation_acc:
					best_validation_acc = validation_acc
					self.models = models

	def train_hmms_no_validation(self):

		n_components = 10
		pseudocount = 1

		for g in self.train_traces_by_goal:
			print(g, len(self.train_traces_by_goal[g]))
			X = self.train_traces_by_goal[g]
			
			self.models[g] = HiddenMarkovModel.from_samples(distribution=DiscreteDistribution, n_components=n_components, X=X, algorithm='baum-welch', pseudocount=pseudocount, verbose=False, stop_threshold=1e-3, max_iterations=1e6)
			self.models[g].bake(merge="None")

	def validation(self, models):
		acc = 0 
		for (tau, goal) in zip(self.validation_traces, self.validation_labels):
			acc += self.eval_trace(tau, goal, models)

		return acc / len(self.validation_traces)


	## We need to manually transform the sequence because of a bug in pomegranate
	def transform_seq(self, seq, keymap):
		output = numpy.empty(len(seq), dtype=numpy.float64)
		for i in range(len(seq)):
			if seq[i] in keymap:
				output[i] = keymap[seq[i]]
			else:
				output[i] = -1
		return output 

	def eval_trace(self, tau, g_true, models = None):
		max_logp = -100000000000
		g_pred = None 

		if models is None:
			models = self.models

		for g in self.models:
			keymap = models[g].keymap[0]
			logp = models[g].log_probability(self.transform_seq(tau, keymap), check_input=False) 
			# print(g, logp)
			if logp > max_logp:
				max_logp = logp 
				g_pred = g

		if g_pred == g_true:
			return 1
		else:
			return 0

	def evaluate(self, output_to_file = False):
		max_len = max([len(tau) for tau in self.test_traces])
		convergence_acc = [0 for i in range(max_len)]
		percent_acc = [0] * 5
		pcts = [0.2, 0.4, 0.6, 0.8, 1]
		
		for (tau, goal) in zip(self.test_traces, self.test_labels):
			## Convergence accuracy
			for t in range(len(tau)):
				result = self.eval_trace(tau[:t+1], goal)
				convergence_acc[t] += result 

				if t == len(tau)-1:
					for t2 in range(len(tau), max_len):
						convergence_acc[t2] += result

			## Percent accuracy
			for i in range(len(pcts)):
				length = max(1, int(pcts[i] * len(tau)))
				result = self.eval_trace(tau[:length], goal)
				percent_acc[i] += result


		for i in range(len(convergence_acc)):
			convergence_acc[i] /= len(self.test_traces)

		for i in range(len(percent_acc)):
			percent_acc[i] /= len(self.test_traces)

		print(", ".join(map(str,convergence_acc)))
		print(", ".join(map(str,percent_acc)))

		if output_to_file:
			outf = open("traces/" + base + "/results/hmm_" + str(iteration) + ".txt", "w")
			outf.write(", ".join(map(str,convergence_acc)) + "; " + ", ".join(map(str,percent_acc)) + "; " + "0")

		return convergence_acc, percent_acc

base = "mit_1"
for iteration in range(1, 2):
	hmm_model = HMM(base, iteration)
	if VALIDATION:
		hmm_model.train_hmms()
	else:
		hmm_model.train_hmms_no_validation()
	hmm_model.evaluate(output_to_file = True)
