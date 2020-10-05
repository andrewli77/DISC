from random import *

input_dir = "kitchen_multigoal/"
base_file = "full.txt"
n_instances = 30

data = open(input_dir + base_file).read().strip().split("\n")
traces_by_goal = {}

for line in data:
	t,g = line.split(";")

	if g not in traces_by_goal:
		traces_by_goal[g] = []
	traces_by_goal[g].append(line)

for file_i in range(n_instances):
	train_traces = []
	test_traces = []

	for g in traces_by_goal:
		shuffle(traces_by_goal[g])

		n_test = max(1, int(0.2 * len(traces_by_goal[g])))

		for i in range(n_test):
			test_traces.append(traces_by_goal[g][i])

		for i in range(n_test, len(traces_by_goal[g])):
			train_traces.append(traces_by_goal[g][i])

	shuffle(train_traces)
	shuffle(test_traces)

	train_outfile = open(input_dir + "train_" + str(file_i+1) + ".txt", "w")
	test_outfile = open(input_dir + "test_" + str(file_i+1) + ".txt", "w")

	for trace in train_traces:
		train_outfile.write(trace + "\n")

	for trace in test_traces:
		test_outfile.write(trace + "\n")

	train_outfile.close()
	test_outfile.close()