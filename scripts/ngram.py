from random import shuffle, choice
import copy

"""
n-gram models use Bayes' rule to find the goal that maximizes P(goal | obs_1, ..., obs_k)
=> G* = argmax P(obs_1,...,obs_k | goal) * P(goal)
n-gram assumes P(obs_i | obs_1,...,obs_{i-1}, goal) = P(obs_i | obs_{i-n+1}, obs_{i-1}, goal)
for computational tractability. The argmax can be computed using the Chain rule.  
"""
class OneGramModel():
	def __init__(self, train_file, test_file, alpha = 0.5):
		self.alpha = alpha # Smoothing constant
		self.train_traces = []
		self.test_traces = []
		self.goals = set()
		self.obs = set()

		# freq[(o,g)] => Number of occurrences of o under goal g
		self.freq = dict()

		# count[g] => Total number of observations under goal g
		self.count = dict()

		# p_goal[g] = P[g]
		self.p_goal = dict()

		self.parse_train_test(train_file, test_file)
		self.initialize_priors()

	def train(self):
		goal_count = 0
		for (tau, goal) in self.train_traces:
			self.incr(self.count, goal, len(tau))
			self.incr(self.p_goal, goal, 1)
			goal_count += 1

			for o in tau:
				self.incr(self.freq, (o,goal), 1)

		# Normalize p_goals
		for goal in self.p_goal:
			self.p_goal[goal] /= goal_count

	"""
	The following two functions are only used for early prediction experiments
	"""
	def utility(self, t):
		return max(1 - t/40, 0)

	def early_test(self):
		score = 0
		time_percent = 0

		for (tau, goal) in self.test_traces:
			output = self.eval(tau)
			max_utility = -1
			prediction_g = None
			prediction_t = None

			for t in range(1, len(output)+1):
				max_p = max([output[t-1][g] for g in self.goals])
				predicted_g = choice([g for g in self.goals if output[t-1][g] == max_p])
				expected_utility = self.utility(t) * output[t-1][predicted_g]

				if expected_utility > max_utility:
					max_utility = expected_utility
					prediction_g = predicted_g
					prediction_t = t

			if prediction_g == goal:
				score += self.utility(prediction_t)
				time_percent += prediction_t / len(tau)


		return score / len(self.test_traces)

	def test(self):
		correct = 0 
		total = 0

		for (tau, goal) in self.test_traces:
			output = self.eval(tau)
			correct += output[-1][goal]
			total += 1
		return float(correct) / total

	def convergence_test(self, distr=False):
		"""
		This function returns the accuracy given t timesteps, for every t up to the max length of a trace
		"""
		max_len = max([len(tau) for (tau,_) in self.test_traces])
		convergence_acc = [0 for i in range(max_len)]

		for (tau, goal) in self.test_traces:
			output = self.eval(tau)
			for i in range(len(output)):
				if distr:
					convergence_acc[i] += output[i][goal]
				else:
					max_p = max([output[i][g] for g in self.goals])
					g_pred = choice([g for g in self.goals if output[i][g] == max_p])
					if g_pred == goal:
						convergence_acc[i] += 1

			if not distr:
				max_p = max([output[-1][g] for g in self.goals])
				g_pred = choice([g for g in self.goals if output[-1][g] == max_p])

			for i in range(len(output), max_len):
				if distr:
					convergence_acc[i] += output[-1][goal]
				else:
					if g_pred == goal:
						convergence_acc[i] += 1

		for i in range(len(convergence_acc)):
			convergence_acc[i] /= len(self.test_traces)

		return convergence_acc

	def convergence_percent_test(self):
		"""
		Returns accuracy given 20%, 40%, 60%, 80%, 100% of full traces
		"""
		convergence_acc = [0] * 5
		pcts = [0.2, 0.4, 0.6, 0.8, 1]

		for (tau, goal) in self.test_traces:
			output = self.eval(tau)

			for i in range(len(pcts)):
				idx = max(1, int(pcts[i] * len(tau)))
				max_p = max([output[idx-1][g] for g in self.goals])
				g_pred = choice([g for g in self.goals if output[idx-1][g] == max_p])

				if g_pred == goal:
					convergence_acc[i] += 1

		for i in range(len(convergence_acc)):
			convergence_acc[i] /= len(self.test_traces)

		return convergence_acc

	def eval(self, tau):
		"""
		Given a trace tau, returns a list of length len(tau) with the predicted goal distribution at each stage.
		"""
		lik_goals = dict()
		predicted_goals = []

		# Assign a prior probability of P[goal]
		for goal in self.goals:
			lik_goals[goal] = self.p_goal[goal]

		for i in range(len(tau)):
			o = tau[i]

			for goal in self.goals:
				lik_goals[goal] *= self.conditional(o, goal)

			lik_goals = self.normalize(lik_goals)
			predicted_goals.append(copy.deepcopy(lik_goals))
		return predicted_goals

	def normalize(self, likelihoods):
		total = 0 

		for goal in likelihoods:
			total += likelihoods[goal]

		for goal in likelihoods:
			likelihoods[goal] /= total

		return likelihoods


	def incr(self, d, k, x):
		if k not in d:
			d[k] = x
		else:
			d[k] += x

	def conditional(self, o, g):
		return float(self.freq[(o,g)]) / self.count[g]

	# Initialize a prior distribution using Laplace smoothing with additive constant alpha.
	def initialize_priors(self):
		for g in self.goals:
			for o in self.obs:
				self.freq[(o, g)] = self.alpha
			self.count[g] = self.alpha * len(self.obs)

	def parse_train_test(self, train_file, test_file):
		train_data = open(train_file).read().strip().split("\n")
		test_data = open(test_file).read().strip().split("\n")

		for line in train_data:
			t,g = line.split(";")
			t = t.strip().split(",")
			self.train_traces.append((t, g))
			self.goals.add(g)
			for o in t:
				self.obs.add(o)

		for line in test_data:
			t,g = line.split(";")
			t = t.strip().split(",")
			self.test_traces.append((t, g))
			self.goals.add(g)
			for o in t:
				self.obs.add(o)

		shuffle(self.train_traces)
		shuffle(self.test_traces)

class TwoGramModel():
	def __init__(self, train_file, test_file, alpha=0.5):
		self.alpha = alpha
		self.train_traces = []
		self.test_traces = []
		self.goals = set()
		self.obs = set()

		# freq[(o, lasto ,g)] => Number of occurrences of o under last_o, g
		self.freq = dict()

		# count[o,g] => Total number of observations of o,g
		self.count = dict()

		# p_goal[g] = P[g]
		self.p_goal = dict()

		self.parse_train_test(train_file, test_file)
		self.initialize_priors()


	def train(self):
		goal_count = 0
		for (tau, goal) in self.train_traces:
			self.incr(self.p_goal, goal, 1)
			goal_count += 1

			for i in range(len(tau)):
				last_o = "null" if (i == 0) else tau[i-1]
				self.incr(self.count, (last_o, goal), 1)
				self.incr(self.freq, (tau[i], last_o, goal), 1)
		# Normalize p_goals
		for goal in self.p_goal:
			self.p_goal[goal] /= goal_count

	def utility(self, t):
		return max(1 - t/40, 0)

	def early_test(self):
		score = 0
		time_percent = 0

		for (tau, goal) in self.test_traces:
			output = self.eval(tau)
			max_utility = -1
			prediction_g = None
			prediction_t = None

			for t in range(1, len(output)+1):
				max_p = max([output[t-1][g] for g in self.goals])
				predicted_g = choice([g for g in self.goals if output[t-1][g] == max_p])
				expected_utility = self.utility(t) * output[t-1][predicted_g]

				if expected_utility > max_utility:
					max_utility = expected_utility
					prediction_g = predicted_g
					prediction_t = t

			if prediction_g == goal:
				score += self.utility(prediction_t)
				time_percent += prediction_t / len(tau)


		return score / len(self.test_traces)

	def test(self):
		correct = 0 
		total = 0

		for (tau, goal) in self.test_traces:
			output = self.eval(tau)
			correct += output[-1][goal]
			total += 1
		return float(correct) / total

	def convergence_test(self, distr=False):
		max_len = max([len(tau) for (tau,_) in self.test_traces])
		convergence_acc = [0 for i in range(max_len)]

		for (tau, goal) in self.test_traces:
			output = self.eval(tau)
			for i in range(len(output)):
				if distr:
					convergence_acc[i] += output[i][goal]
				else:
					max_p = max([output[i][g] for g in self.goals])
					g_pred = choice([g for g in self.goals if output[i][g] == max_p])
					if g_pred == goal:
						convergence_acc[i] += 1

			if not distr:
				max_p = max([output[-1][g] for g in self.goals])
				g_pred = choice([g for g in self.goals if output[-1][g] == max_p])

			for i in range(len(output), max_len):
				if distr:
					convergence_acc[i] += output[-1][goal]
				else:
					if g_pred == goal:
						convergence_acc[i] += 1
						
		for i in range(len(convergence_acc)):
			convergence_acc[i] /= len(self.test_traces)

		return convergence_acc

	def convergence_percent_test(self):
		convergence_acc = [0] * 5
		pcts = [0.2, 0.4, 0.6, 0.8, 1]

		for (tau, goal) in self.test_traces:
			output = self.eval(tau)

			for i in range(len(pcts)):
				idx = max(1, int(pcts[i] * len(tau)))
				max_p = max([output[idx-1][g] for g in self.goals])
				g_pred = choice([g for g in self.goals if output[idx-1][g] == max_p])

				if g_pred == goal:
					convergence_acc[i] += 1

		for i in range(len(convergence_acc)):
			convergence_acc[i] /= len(self.test_traces)

		return convergence_acc

	## Given a trace tau, returns a list of length len(tau) with the predicted goal at each stage.
	def eval(self, tau):
		lik_goals = dict()
		predicted_goals = []

		# Assign a prior probability of P[goal]
		for goal in self.goals:
			lik_goals[goal] = self.p_goal[goal]


		for i in range(len(tau)):
			last_o = "null" if (i == 0) else tau[i-1]
			o = tau[i]

			for goal in self.goals:
				lik_goals[goal] *= self.conditional(o, last_o, goal)

			lik_goals = self.normalize(lik_goals)
			predicted_goals.append(copy.deepcopy(lik_goals))
		return predicted_goals

	def normalize(self, likelihoods):
		total = 0 

		for goal in likelihoods:
			total += likelihoods[goal]

		for goal in likelihoods:
			likelihoods[goal] /= total

		return likelihoods

	def incr(self, d, k, x):
		if k not in d:
			d[k] = x
		else:
			d[k] += x

	## Use a Laplace smoothing: assume a frequency of 1 if the observation has never occurred in training. 
	def conditional(self, o, last_o, g):
		return float(self.freq[(o,last_o, g)]) / self.count[last_o, g]

	# Initialize a prior distribution using Laplace smoothing with additive constant alpha.
	def initialize_priors(self):
		for g in self.goals:
			for o_last in list(self.obs) + ["null"]:
				for o in self.obs:
					self.freq[(o, o_last, g)] = self.alpha
				self.count[(o_last, g)] = self.alpha * len(self.obs)

	def parse_train_test(self, train_file, test_file):
		train_data = open(train_file).read().strip().split("\n")
		test_data = open(test_file).read().strip().split("\n")

		for line in train_data:
			t,g = line.split(";")
			t = t.strip().split(",")
			self.train_traces.append((t, g))
			self.goals.add(g)
			for o in t:
				self.obs.add(o)

		for line in test_data:
			t,g = line.split(";")
			t = t.strip().split(",")
			self.test_traces.append((t, g))
			self.goals.add(g)
			for o in t:
				self.obs.add(o)

		shuffle(self.train_traces)
		shuffle(self.test_traces)

base = "traces/alfred"
model_name = "one" # "one" for 1-gram, "two" for 2-gram

for i in range(1, 31):
	if model_name == "one":
		model = OneGramModel(base + "/train_" + str(i) + ".txt", base + "/test_" + str(i) + ".txt")
	else:
		model = TwoGramModel(base + "/train_" + str(i) + ".txt", base + "/test_" + str(i) + ".txt")
	model.train()

	results = model.convergence_test(distr = False)
	pct_results = model.convergence_percent_test()
	early_results = model.early_test()

	print("Convergence results:")
	for each in results:
		print("%.3f, " %(each), end="")
	print("")
	print("Convergence percent results:")
	
	for each in pct_results:
		print("%.3f, " %(each), end="")
	print("")
	print(early_results)

	if model_name == "one":
		outf = open(base + "/results/" + "1gram_" + str(i) + ".txt", "w")
	else:
		outf = open(base + "/results/" + "2gram_" + str(i) + ".txt", "w")

	outf.write(", ".join(map(str, results)) + "; " + ", ".join(map(str, pct_results)) + "; " + str(early_results))
	outf.close()
