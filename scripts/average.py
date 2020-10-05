import statistics 

n = 30
base = "traces/alfred"
#base = "traces/starcraft_action_types_only"
#base = "traces/malware/BOOT_COMPLETED"
baseline = "2gram"

results = []
for i in range(11, n+1):
	f = open(base + "/results/" + baseline + "_" + str(i) + ".txt").read().strip().split(";")
	results.append(float(f[-1]))

avg = sum(results) / len(results)
stderr = statistics.stdev(results) / (len(results) ** 0.5)

print("Average utility:", avg, "--- error:", 1.729 * stderr)
#print("Decision time:", sum(times) / len(times))