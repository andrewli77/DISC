import statistics 

n = 30
base = "traces/starcraft_action_types_only"

def load_from_model(file_name):
	dfas = open(file_name).read().strip().split("----------- DFA ----------- ")[1:]

	num_states = 0
	num_edges = 0
	for dfa in dfas:
		dfa = dfa.split("\n")[1:-2]
		num_edges += len(dfa)
		states = set()
		for transition in dfa:
			from_state = transition[6:transition.find(",")]
			to_state = transition[transition.find("= ")+2:]
			states.add(from_state)
			states.add(to_state)
		
		states.add('0')
		states.add('1')
		states.add('2')
		num_states += len(states)

	return num_states / len(dfas), num_edges / len(dfas)



num_states = 0
num_edges = 0

for i in range(11, n+1):
	f = load_from_model(base + "/results/ours_" + str(i) + "_models.txt")#open(base + "/results/tree_only_" + str(i) + "_sizes.txt").read().strip().split(",")
	num_states += float(f[0])
	num_edges += float(f[1])

print(num_states / 20, num_edges / 20)

#stderr = statistics.stdev(results) / (len(results) ** 0.5)

# print("Average utility:", avg, "--- error:", 1.833 * stderr)
#print("Decision time:", sum(times) / len(times))