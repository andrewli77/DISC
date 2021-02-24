"""
This file uses PyTorch's LSTM implementation for sequence classification.
It's recommended to not use GPU for this. 
"""

import torch
import argparse
from random import shuffle
from numpy import argmax

from GLOBAL_VARS import *

device = torch.device("cpu")


class LSTM_Baseline(torch.nn.Module):

	## h_dim: Dimension of the hidden layer
	def __init__(self, train_file, test_file, optimize_convergence = False, stack_layers = 1, num_epochs = 300):
		super(LSTM_Baseline, self).__init__()
		self.h_dim = 50 # This is not used except for initialization
		self.optimize_convergence = optimize_convergence
		self.stack_layers = stack_layers
		self.num_epochs = num_epochs
		self.train_traces = []
		self.validation_traces = []
		self.test_traces = []

		# We assume actions can be in any format, so here 
		# we use a one-to-one map from actions to {1,...,n_actions} 
		# and encode an action as a one-hot vector.
		self.action_map = dict() 
		self.num_actions = 0 


		# Similarly, assume goals are in any format.
		# We map goals to {1,..., n_goals}

		self.goal_map = dict()
		self.num_goals = 0

		if MULTILABEL:
			self.parse_train_test_multilabel(train_file, test_file)
		else:
			self.parse_train_validation_test(train_file, test_file)

		# Input is a sequence of actions, output is a hidden state. 
		self.lstm = torch.nn.LSTM(self.num_actions, self.h_dim, num_layers = self.stack_layers)
		# Input is a hidden state, output is a goal state. 
		self.hidden_to_goal = torch.nn.Linear(self.h_dim, self.num_goals)

	def forward(self, seq):
		"""
		seq: a tensor of size (seq_length x 1 x num_actions)
		"""
		seq_len = seq.size()[0]
		hidden_states, (lstm_out,_) = self.lstm(seq)
		goal_out = self.hidden_to_goal(hidden_states)

		if MULTILABEL:
			return goal_out.view(seq_len, self.num_goals)
		else:
			goal_scores = torch.nn.functional.log_softmax(goal_out, dim=2)
			return goal_scores.view(seq_len, self.num_goals)

	def train(self):
		best_validation_acc = -1
		best_convergence = None
		best_percent_convergence = None
		best_early_utility = None

		if VALIDATION:
			batch_params = [8, 32]
			h_dims = [25, 50]
		else:
			batch_params = [8]
			h_dims = [25]

		for batch_size in batch_params:
			for h_dim in h_dims:
				print(batch_size, h_dim)
				self.lstm = torch.nn.LSTM(self.num_actions, h_dim, num_layers = self.stack_layers)
				self.hidden_to_goal = torch.nn.Linear(h_dim, self.num_goals)
				self.to(device)

				if MULTILABEL:
					loss_function = torch.nn.BCEWithLogitsLoss(reduction="mean")
				else:
					loss_function = torch.nn.NLLLoss(reduction="mean")
				optimizer = torch.optim.Adam(self.parameters())

				for epoch in range(self.num_epochs):
					print(epoch)
					total_loss = 0
					predictive_acc = 0
					self.zero_grad()

					for i in range(len(self.train_traces)):
						tau,goal = self.train_traces[i]

						output = self(tau)

						
						if self.optimize_convergence:
							if MULTILABEL:
								target = torch.cat(len(output)*[goal.unsqueeze(0)])
								loss = loss_function(output, target)
							else:
								# Average predictive accuracy over entire trace
								loss = loss_function(output, torch.tensor([goal for i in range(len(output))], device=device, dtype=torch.long))


						else:
							assert(not MULTILABEL)
							# Predictive accuracy of the full traces
							loss = loss_function(output[-1].view(1, -1), torch.tensor([goal], device=device, dtype=torch.long))


						loss.backward()

						if (i+1) % batch_size == 0 or i == len(self.train_traces) - 1:
							optimizer.step()
							self.zero_grad()

						total_loss += loss

					if VALIDATION:
						valid_acc = self.test(validation=True)

					test_acc = self.test()
					predictive_acc /= len(self.train_traces)
					print("Loss: %.3f, Training accuracy: %.3f, Valid acc: %.3f, Test acc: %.3f" %(total_loss, predictive_acc, -1, test_acc))

					if VALIDATION and valid_acc >= best_validation_acc:
						best_validation_acc = valid_acc
						best_convergence = self.test_convergence()
						best_percent_convergence = self.test_percent_convergence()
						best_early_utility = self.early_test()	

		if VALIDATION:
			return best_convergence, best_percent_convergence, best_early_utility
		elif MULTILABEL:
			return self.test_convergence_multilabel(), None, None
		else:
			return self.test_convergence(), self.test_percent_convergence(), self.early_test()

	"""
	The following two functions are only used for early prediction experiments.
	"""

	def utility(self, t):
		return max(1 - t/40, 0)

	def early_test(self):
		with torch.no_grad():
			score = 0
			time_percent = 0

			for (tau, goal) in self.test_traces:
				output = self(tau).cpu()

				max_utility = -1
				prediction_g = None
				prediction_t = None

				for t in range(1, len(output)+1):
					predicted_g =  argmax(output[t-1]).item()
					expected_utility = self.utility(t) * torch.exp(output[t-1][predicted_g]).item()
					if expected_utility > max_utility:
						max_utility = expected_utility
						prediction_g = predicted_g
						prediction_t = t

				if prediction_g == goal:
					#print(prediction_t, len(tau))
					score += self.utility(prediction_t)
					time_percent += prediction_t / len(tau)

			return score / len(self.test_traces) , time_percent / len(self.test_traces)



	def test(self, validation=False, output_probs=False):
		with torch.no_grad():
			most_prob_accuracy = 0 # The accuracy if we choose choose a single goal based on highest probability
			true_accuracy = 0	# The accuracy of the underlying output distribution

			if validation:
				assert(VALIDATION)
				for (tau, goal) in self.validation_traces:
					output = self(tau).cpu()
					most_prob_accuracy += (argmax(output[-1]).item() == goal) 

				return (most_prob_accuracy / len(self.validation_traces))
			else:
				if MULTILABEL:
					for (tau, goal) in self.test_traces:
						output = self(tau).cpu()

						for i in range(self.num_goals):
							if output[-1][i].item() > 0:
								if goal[i].item() == 1:
									true_accuracy += 1
							elif output[-1][i].item() <= 0:
								if goal[i].item() == 0:
									true_accuracy += 1

					return (true_accuracy / len(self.test_traces) / self.num_goals)
				else:
					for (tau, goal) in self.test_traces:
						output = self(tau).cpu()
						if output_probs:
							print(output[-1][goal].item())
						most_prob_accuracy += (argmax(output[-1]).item() == goal) 

					return (most_prob_accuracy / len(self.test_traces))

	def test_convergence(self):
		"""
		Computes the convergence accuracy of the LSTM on the test set. Returns an array of accuracies where index i is the 
		overall accuracy after i observations of the trace are shown. If a trace has less than i observations, we use the final classification.
		Here we consider only the accuracy of the probability distribution rather than the "highest probability classifier"
		"""

		max_len = max([tau.size()[0] for (tau,_) in self.test_traces])
		convergence_acc = [0 for i in range(max_len)]

		with torch.no_grad():
			for (tau, goal) in self.test_traces:
				output = self(tau).cpu()
				for i in range(len(tau)):

					probs = [output[i][g].item() for g in range(self.num_goals)]
					convergence_acc[i] += (argmax(probs) == goal) 

					if i == len(tau)-1:
						for j in range(len(tau), max_len):
							convergence_acc[j] += (argmax(probs) == goal) 

			# Normalize the probabilities
			for i in range(len(convergence_acc)):
				convergence_acc[i] /= len(self.test_traces)

			return convergence_acc

	def test_convergence_multilabel(self):
		assert(MULTILABEL)

		max_len = max([tau.size()[0] for (tau,_) in self.test_traces])
		convergence_acc = [0 for i in range(max_len)]

		with torch.no_grad():
			for (tau, goal) in self.test_traces:
				output = self(tau).cpu()
				for i in range(len(tau)):
					sc = 0
					for j in range(self.num_goals):
						if output[i][j].item() > 0:
							if goal[j].item() == 1:
								sc += 1
						elif output[i][j].item() <= 0:
							if goal[j].item() == 0:
								sc += 1

					convergence_acc[i] += sc

					if i == len(tau)-1:
						for k in range(len(tau), max_len):
							convergence_acc[k] += sc

			# Normalize the probabilities
			for i in range(len(convergence_acc)):
				convergence_acc[i] /= len(self.test_traces) * self.num_goals

			return convergence_acc

	def test_percent_convergence(self):
		"""
		Returns accuracy given 20%, 40%, 60%, 80%, and 100% of the full trace lengths
		"""
		convergence_acc = [0,0,0,0,0]

		with torch.no_grad():
			for (tau, goal) in self.test_traces:
				output = self(tau).cpu()
				pct = [0.2, 0.4, 0.6, 0.8, 1]
				for i in range(5):
					idx = max(1, int(pct[i] * len(tau)))
					probs = [output[idx-1][g].item() for g in range(self.num_goals)]
					convergence_acc[i] += (argmax(probs) == goal) 

			for i in range(len(convergence_acc)):
				convergence_acc[i] /= len(self.test_traces)

			return convergence_acc

	def parse_train_test_multilabel(self, train_file, test_file):
		assert(MULTILABEL)

		train_data = open(train_file).read().strip().split("\n")
		test_data = open(test_file).read().strip().split("\n")
		tmp_train_traces = []
		tmp_test_traces = []


		for line in train_data:
			t, gs = line.split(";")
			t = t.strip().split(",")
			gs = gs.strip().split(",")
			tau = []
			goals = []
			# Here we assign the action mapping from actions to {1,..., n_actions}
			for o in t:
				if o not in self.action_map:
					self.action_map[o] = self.num_actions
					self.num_actions += 1

				action_idx = self.action_map[o]
				tau.append(action_idx)

			for g in gs:
				if g not in self.goal_map:
					self.goal_map[g] = self.num_goals
					self.num_goals += 1
				goal = self.goal_map[g]
				goals.append(goal)

			tmp_train_traces.append((tau, goals))

		for line in test_data:
			t, gs = line.split(";")
			t = t.strip().split(",")
			gs = gs.strip().split(",")
			tau = []
			goals = []
			# Here we assign the action mapping from actions to {1,..., n_actions}
			for o in t:
				if o not in self.action_map:
					self.action_map[o] = self.num_actions
					self.num_actions += 1

				action_idx = self.action_map[o]
				tau.append(action_idx)

			for g in gs:
				if g not in self.goal_map:
					self.goal_map[g] = self.num_goals
					self.num_goals += 1
				goal = self.goal_map[g]
				goals.append(goal)

			tmp_test_traces.append((tau, goals))


		# Finally, we convert the data into tensors of the proper shapes.

		for (tau, goals) in tmp_train_traces:
			new_tau = []
			for a in tau:
				new_tau.append(self.action_to_one_hot(a).view(1,1,-1))

			tau_tensor = torch.cat(new_tau).to(device=device)

			goal_tensor = torch.zeros(self.num_goals)
			for goal in goals:
				goal_tensor[goal] = 1

			self.train_traces.append((tau_tensor, goal_tensor))


		for (tau, goals) in tmp_test_traces:
			new_tau = []
			for a in tau:
				new_tau.append(self.action_to_one_hot(a).view(1,1,-1))

			goal_tensor = torch.zeros(self.num_goals)
			for goal in goals:
				goal_tensor[goal] = 1

			tau_tensor = torch.cat(new_tau).to(device=device)
			self.test_traces.append((tau_tensor, goal_tensor))


	def parse_train_validation_test(self, train_file, test_file):
		"""
		Parses the training and test data into a usable format. Actions are encoded as one-hot vectors.
		"""
		train_data = open(train_file).read().strip().split("\n")
		test_data = open(test_file).read().strip().split("\n")
		tmp_train_traces = []
		tmp_validation_traces = []
		tmp_test_traces = []

		traces_by_goal = {}

		for line in train_data:
			t, g = line.split(";")
			t = t.strip().split(",")
			tau = []

			# Here we assign the action mapping from actions to {1,..., n_actions}
			for o in t:
				if o not in self.action_map:
					self.action_map[o] = self.num_actions
					self.num_actions += 1

				action_idx = self.action_map[o]
				tau.append(action_idx)

			if g not in self.goal_map:	
				self.goal_map[g] = self.num_goals
				self.num_goals += 1
				traces_by_goal[self.goal_map[g]] = []
			goal = self.goal_map[g]
			traces_by_goal[goal].append(tau)

		for goal in traces_by_goal:
			if VALIDATION:
				n_validation = max(int(len(traces_by_goal[goal]) * 0.2), 1)
			else:
				n_validation = 0 

			for i in range(n_validation):
				tmp_validation_traces.append((traces_by_goal[goal][i], goal))
			for i in range(n_validation, len(traces_by_goal[goal])):
				tmp_train_traces.append((traces_by_goal[goal][i], goal))

		for line in test_data:
			t, g = line.split(";")
			t = t.strip().split(",")
			tau = []

			# Here we assign the action mapping from actions to {1,..., n_actions}
			for o in t:
				if o not in self.action_map:
					self.action_map[o] = self.num_actions
					self.num_actions += 1

				action_idx = self.action_map[o]
				tau.append(action_idx)

			if g not in self.goal_map:
				self.goal_map[g] = self.num_goals
				self.num_goals += 1
			goal = self.goal_map[g]

			tmp_test_traces.append((tau, goal))

		# Finally, we convert the data into tensors of the proper shapes.
		for (tau, goal) in tmp_train_traces:
			new_tau = []
			for a in tau:
				new_tau.append(self.action_to_one_hot(a).view(1,1,-1))

			tau_tensor = torch.cat(new_tau).to(device=device)
			self.train_traces.append((tau_tensor, int(goal)))

		for (tau, goal) in tmp_validation_traces:
			new_tau = []
			for a in tau:
				new_tau.append(self.action_to_one_hot(a).view(1,1,-1))

			tau_tensor = torch.cat(new_tau).to(device=device)
			self.validation_traces.append((tau_tensor, int(goal)))

		for (tau, goal) in tmp_test_traces:
			new_tau = []
			for a in tau:
				new_tau.append(self.action_to_one_hot(a).view(1,1,-1))

			tau_tensor = torch.cat(new_tau).to(device=device)
			self.test_traces.append((tau_tensor, int(goal)))



	def action_to_one_hot(self,action_idx):
		a = torch.zeros(self.num_actions)
		a[action_idx] = 1
		return a


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='e.g. alfred, crystal')
parser.add_argument('--ids', type=int, nargs='+')
parser.add_argument('--c', action="store_true", default=False, help='specify for convergence optimization')
args = parser.parse_args()


optimize_convergence = args.c 
stack_layers = 2
num_epochs = 10
base_names = ["traces/" + args.dataset]
ids = args.ids

for base_name in base_names:
	print(base_name)
	for i in ids:
		train_f = base_name + "/train_" + str(i) + ".txt"
		test_f = base_name + "/test_" + str(i) + ".txt"
		print(train_f, i)

		lstm_baseline = LSTM_Baseline(train_f, test_f, optimize_convergence=optimize_convergence, stack_layers=stack_layers, num_epochs=num_epochs)

		lstm_baseline.to(device)
		results = lstm_baseline.train()

		if MULTILABEL:
			print(train_f)
			print("Optimize convergence: ", optimize_convergence)
			print("Convergence:", results[0])

			idtfr = "lstm_" + ("c_" if optimize_convergence else "") + str(i)
			output_name = base_name + "/results/" + idtfr + ".txt"

			output_f = open(output_name, "w")
			output_f.write(", ".join(map(str, results[0])))
			output_f.close()
		else:
			print(train_f)
			print("Optimize convergence: ", optimize_convergence)
			print("Convergence:", results[0])
			print("Percent convergence:", results[1])
			print("Early utility:", results[2][0])
			print("Early decision time:", results[2][1])

			idtfr = "lstm_" + ("c_" if optimize_convergence else "") + str(i)
			output_name = base_name + "/results/" + idtfr + ".txt"

			output_f = open(output_name, "w")
			output_f.write(", ".join(map(str, results[0])) + "; ")
			output_f.write(", ".join(map(str, results[1])) + "; ")
			output_f.write(str(results[2][0]))
			output_f.close()
