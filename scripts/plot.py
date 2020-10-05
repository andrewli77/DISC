import matplotlib.pyplot as plt 
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rc
import numpy as np
import statistics

plt.rcParams["font.family"] = "Times New Roman"

def parse_and_avg(file_prefix, num_files=1):
	collection_1 = []
	collection_2 = []
	output_1 = []
	output_1_lower = []
	output_1_upper = []

	output_2 = [] 
	output_2_error = []

	for i in range(num_files):
		f = open(file_prefix + "_" + str(i+1) + ".txt").read().split(";")
		f1 = f[0].strip().replace("'", "").split(", ")
		f2 = f[1].strip().replace("'", "").split(", ")
		collection_1.append(list(map(float, f1)))
		collection_2.append(list(map(float, f2)))

	max_len = max([len(collection_1[i]) for i in range(len(collection_1))])

	for i in range(max_len):
		accs = []

		for j in range(len(collection_1)):
			if i >= len(collection_1[j]):
				accs.append(collection_1[j][-1])
			else:
				accs.append(collection_1[j][i])

		avg = sum(accs) / len(accs)
		stderr = statistics.stdev(accs) / (len(accs) ** 0.5)

		output_1.append(avg)
		output_1_lower.append(avg-1.699*stderr)
		output_1_upper.append(avg+1.699*stderr)

	max_len = max([len(collection_2[i]) for i in range(len(collection_2))])
	for i in range(len(collection_2[0])):
		accs = []

		for j in range(len(collection_2)):
			if i >= len(collection_2[j]):
				accs.append(collection_2[j][-1])
			else:
				accs.append(collection_2[j][i])

		avg = sum(accs) / len(accs)
		stderr = statistics.stdev(accs) / (len(accs) ** 0.5)
		output_2.append(avg)
		output_2_error.append(1.699*stderr)

	return (np.array(output_1),np.array(output_1_lower), np.array(output_1_upper)) , (np.array(output_2), np.array(output_2_error))

def parse_and_avg_kitchen(file_prefix, num_files=1):
	collection_1 = []

	output_1 = []
	output_1_lower = []
	output_1_upper = []


	for i in range(num_files):
		f = open(file_prefix + "_" + str(i+1) + ".txt").read().strip()
		f1 = f.replace("'", "").split(", ")
		collection_1.append(list(map(float, f1)))

	max_len = max([len(collection_1[i]) for i in range(len(collection_1))])

	for i in range(max_len):
		accs = []

		for j in range(len(collection_1)):
			if i >= len(collection_1[j]):
				accs.append(collection_1[j][-1])
			else:
				accs.append(collection_1[j][i])

		avg = sum(accs) / len(accs)
		stderr = statistics.stdev(accs) / (len(accs) ** 0.5)

		output_1.append(avg)
		output_1_lower.append(avg-1.699*stderr)
		output_1_upper.append(avg+1.699*stderr)

	return np.array(output_1),np.array(output_1_lower), np.array(output_1_upper)


def make_plot_kitchen():
	ours = parse_and_avg_kitchen("traces/kitchen_multigoal/results/ours", num_files=30)
	lstm = parse_and_avg_kitchen("traces/kitchen_multigoal/results/lstm_c", num_files=30)


	fig, ax1 = plt.subplots(1, 1)
	plt.subplots_adjust(top = 0.92, bottom = 0.23, hspace = 0, wspace = 0, left=0.07, right = 0.96)
	fig.set_size_inches(10,4)

	ax1.set_xscale("log", basex = 2)
	ax1.set_ylabel("Accuracy", fontsize = 16, labelpad=10)
	ax1.tick_params(labelsize=12)
	ax1.set_title("Kitchen Multigoal", fontsize = 16)
	ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

	L = len(ours[0][1:])
	xrange = range(1, L+1)


	ax1.plot(xrange, ours[0][1:], label="DISC (ours)",linestyle=(0, (5,1)), color="b")
	ax1.fill_between(xrange, ours[1][1:], ours[2][1:], color="b", alpha=0.2)

	ax1.plot(xrange, lstm[0], label="LSTM",  color="r")
	ax1.fill_between(xrange, lstm[1], lstm[2], color="r", alpha=0.2)

	ax1.set_ylim((0,1))

	fig.text(0.5, 0.15, 'Number of observations', ha='center', fontsize = 16)

	handles, labels = ax1.get_legend_handles_labels()
	legend = fig.legend(handles, labels, loc="lower center",bbox_to_anchor = (0.5, 0.03), markerscale=6, fontsize=16, ncol = 6)

	for i in range(len(legend.get_lines())):
		legend.get_lines()[i].set_linewidth(4)
	plt.show()

def make_plot_double():
	file_prefix_1 = "traces/starcraft_action_types_only"
	file_prefix_2 = "traces/mit_1"
	file_prefix_3 = "traces/crystal"
	file_prefix_4 = "traces/malware/BATTERY_LOW"

	ours_1 = parse_and_avg(file_prefix_1 + "/results/ours", num_files=30)
	tree_only_1 = parse_and_avg(file_prefix_1 + "/results/tree_only", num_files=30)
	lstm_conv_1 = parse_and_avg(file_prefix_1 + "/results/lstm_c", num_files=30)
	hmm_1 = parse_and_avg(file_prefix_1 + "/results/hmm", num_files=30)
	_1g_1 = parse_and_avg(file_prefix_1 + "/results/1gram", num_files=30)
	_2g_1 = parse_and_avg(file_prefix_1 + "/results/2gram", num_files=30)

	ours_2 = parse_and_avg(file_prefix_2 + "/results/ours", num_files=30)
	tree_only_2 = parse_and_avg(file_prefix_2 + "/results/tree_only", num_files=30)
	lstm_conv_2 = parse_and_avg(file_prefix_2 + "/results/lstm_c", num_files=30)
	hmm_2 = parse_and_avg(file_prefix_2 + "/results/hmm", num_files=30)
	_1g_2 = parse_and_avg(file_prefix_2 + "/results/1gram", num_files=30)
	_2g_2 = parse_and_avg(file_prefix_2 + "/results/2gram", num_files=30)


	ours_3 = parse_and_avg(file_prefix_3 + "/results/ours", num_files=30)
	tree_only_3 = parse_and_avg(file_prefix_3 + "/results/tree_only", num_files=30)
	lstm_conv_3 = parse_and_avg(file_prefix_3 + "/results/lstm_c", num_files=30)
	hmm_3 = parse_and_avg(file_prefix_3 + "/results/hmm", num_files=30)
	_1g_3 = parse_and_avg(file_prefix_3 + "/results/1gram", num_files=30)
	_2g_3 = parse_and_avg(file_prefix_3 + "/results/2gram", num_files=30)

	ours_4 = parse_and_avg(file_prefix_4 + "/results/ours", num_files=30)
	tree_only_4 = parse_and_avg(file_prefix_4 + "/results/tree_only", num_files=30)
	lstm_conv_4 = parse_and_avg(file_prefix_4 + "/results/lstm_c", num_files=30)
	hmm_4 = parse_and_avg(file_prefix_4 + "/results/hmm", num_files=30)
	_1g_4 = parse_and_avg(file_prefix_4 + "/results/1gram", num_files=30)
	_2g_4 = parse_and_avg(file_prefix_4 + "/results/2gram", num_files=30)




	fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
	plt.subplots_adjust(top = 0.92, bottom = 0.28, hspace = 0, wspace = 0, left=0.07, right = 0.96)
	fig.set_size_inches(10,4)

	ax1.set_xscale("log", basex = 2)
	ax1.set_ylabel("Accuracy", fontsize = 16, labelpad=10)
	ax1.tick_params(labelsize=12)
	ax1.set_title("StarCraft", fontsize = 16)
	ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

	L = len(ours_1[0][0][1:])
	xrange = range(1, L+1)


	ax1.plot(xrange, ours_1[0][0][1:], label="DISC (ours)",linestyle=(0, (5,1)), color="b")
	ax1.fill_between(xrange, ours_1[0][1][1:], ours_1[0][2][1:], color="b", alpha=0.2)

	ax1.plot(xrange, tree_only_1[0][0][1:], label="DFA-FT", color="purple")
	ax1.fill_between(xrange, tree_only_1[0][1][1:], tree_only_1[0][2][1:], color="purple", alpha=0.2)

	ax1.plot(xrange, lstm_conv_1[0][0], label="LSTM",  color="r")
	ax1.fill_between(xrange, lstm_conv_1[0][1], lstm_conv_1[0][2], color="r", alpha=0.2)

	ax1.plot(xrange, hmm_1[0][0], label="HMM", color="gray")
	ax1.fill_between(xrange, hmm_1[0][1], hmm_1[0][2], color="gray", alpha=0.2)

	ax1.plot(xrange, _1g_1[0][0], label="One-gram", color="y")
	ax1.fill_between(xrange, _1g_1[0][1], _1g_1[0][2], color="y", alpha=0.2)

	ax1.plot(xrange, _2g_1[0][0], label="Two-gram", color="g")
	ax1.fill_between(xrange, _2g_1[0][1], _2g_1[0][2], color="g", alpha=0.2)

	
	#plt.title(plot_title)
	ax1.set_ylim((0,1))
	#plt.show()

	#plt.clf()
	ax2.set_xscale("log", basex = 2)
	ax2.set_title("MIT-AR", fontsize = 16)
	ax2.tick_params(labelsize=12)
	ax2.yaxis.set_ticklabels([])
	ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))

	L = len(ours_2[0][0][1:])
	xrange = range(1, L+1)

	ax2.plot(xrange, ours_2[0][0][1:], label="DISC (ours)",linestyle=(0, (5,1)), color="b")
	ax2.fill_between(xrange, ours_2[0][1][1:], ours_2[0][2][1:], color="b", alpha=0.2)

	ax2.plot(xrange, tree_only_2[0][0][1:], label="DFA-FT", color="purple")
	ax2.fill_between(xrange, tree_only_2[0][1][1:], tree_only_2[0][2][1:], color="purple", alpha=0.2)

	ax2.plot(xrange, lstm_conv_2[0][0], label="LSTM",  color="r")
	ax2.fill_between(xrange, lstm_conv_2[0][1], lstm_conv_2[0][2], color="r", alpha=0.2)

	ax2.plot(xrange, hmm_2[0][0], label="HMM", color="gray")
	ax2.fill_between(xrange, hmm_2[0][1], hmm_2[0][2], color="gray", alpha=0.2)

	ax2.plot(xrange, _1g_2[0][0], label="One-gram", color="y")
	ax2.fill_between(xrange, _1g_2[0][1], _1g_2[0][2], color="y", alpha=0.2)

	ax2.plot(xrange, _2g_2[0][0], label="Two-gram", color="g")
	ax2.fill_between(xrange, _2g_2[0][1], _2g_2[0][2], color="g", alpha=0.2)
	#plt.title(plot_title)
	#plt.ylim((0,1))
	ax2.set_ylim((0,1))

	ax3.set_xscale("log", basex = 2)
	ax3.set_title("Crystal Island", fontsize = 16)
	ax3.tick_params(labelsize=12)
	ax3.yaxis.set_ticklabels([])
	ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))

	L = len(ours_3[0][0][1:])
	xrange = range(1, L+1)

	ax3.plot(xrange, ours_3[0][0][1:], label="DISC (ours)",linestyle=(0, (5,1)), color="b")
	ax3.fill_between(xrange, ours_3[0][1][1:], ours_3[0][2][1:], color="b", alpha=0.2)

	ax3.plot(xrange, tree_only_3[0][0][1:], label="DFA-FT", color="purple")
	ax3.fill_between(xrange, tree_only_3[0][1][1:], tree_only_3[0][2][1:], color="purple", alpha=0.2)

	ax3.plot(xrange, lstm_conv_3[0][0], label="LSTM",  color="r")
	ax3.fill_between(xrange, lstm_conv_3[0][1], lstm_conv_3[0][2], color="r", alpha=0.2)

	ax3.plot(xrange, hmm_3[0][0], label="HMM", color="gray")
	ax3.fill_between(xrange, hmm_3[0][1], hmm_3[0][2], color="gray", alpha=0.2)

	ax3.plot(xrange, _1g_3[0][0], label="One-gram", color="y")
	ax3.fill_between(xrange, _1g_3[0][1], _1g_3[0][2], color="y", alpha=0.2)

	ax3.plot(xrange, _2g_3[0][0], label="Two-gram", color="g")
	ax3.fill_between(xrange, _2g_3[0][1], _2g_3[0][2], color="g", alpha=0.2)
	#plt.title(plot_title)
	#plt.ylim((0,1))
	ax3.set_ylim((0,1))


	ax4.set_xscale("log", basex = 2)
	ax4.set_title("BatteryLow", fontsize = 16)
	ax4.tick_params(labelsize=12)
	ax4.yaxis.set_ticklabels([])
	ax4.xaxis.set_major_formatter(FormatStrFormatter('%d'))

	L = len(ours_4[0][0][1:])
	xrange = range(1, L+1)

	ax4.plot(xrange, ours_4[0][0][1:], label="DISC (ours)",linestyle=(0, (5,1)), color="b")
	ax4.fill_between(xrange, ours_4[0][1][1:], ours_4[0][2][1:], color="b", alpha=0.2)

	ax4.plot(xrange, tree_only_4[0][0][1:], label="DFA-FT", color="purple")
	ax4.fill_between(xrange, tree_only_4[0][1][1:], tree_only_4[0][2][1:], color="purple", alpha=0.2)

	ax4.plot(xrange, lstm_conv_4[0][0], label="LSTM",  color="r")
	ax4.fill_between(xrange, lstm_conv_4[0][1], lstm_conv_4[0][2], color="r", alpha=0.2)

	ax4.plot(xrange, hmm_4[0][0], label="HMM", color="gray")
	ax4.fill_between(xrange, hmm_4[0][1], hmm_4[0][2], color="gray", alpha=0.2)

	ax4.plot(xrange, _1g_4[0][0], label="One-gram", color="y")
	ax4.fill_between(xrange, _1g_4[0][1], _1g_4[0][2], color="y", alpha=0.2)

	ax4.plot(xrange, _2g_4[0][0], label="Two-gram", color="g")
	ax4.fill_between(xrange, _2g_4[0][1], _2g_4[0][2], color="g", alpha=0.2)
	#plt.title(plot_title)
	#plt.ylim((0,1))
	ax4.set_ylim((0,1))


	fig.text(0.5, 0.15, 'Number of observations', ha='center', fontsize = 16)

	handles, labels = ax1.get_legend_handles_labels()
	legend = fig.legend(handles, labels, loc="lower center",bbox_to_anchor = (0.5, 0), markerscale=6, fontsize=16, ncol = 6)

	for i in range(len(legend.get_lines())):
		legend.get_lines()[i].set_linewidth(4)
	plt.show()


def make_plots(file_prefix, plot_title, legend=True):
	ours = parse_and_avg(file_prefix + "/results/ours", num_files=30)
	tree_only = parse_and_avg(file_prefix + "/results/tree_only", num_files=30)
	lstm_conv = parse_and_avg(file_prefix + "/results/lstm_c", num_files=30)
	hmm = parse_and_avg(file_prefix + "/results/hmm", num_files=30)
	_1g = parse_and_avg(file_prefix + "/results/1gram", num_files=30)
	_2g = parse_and_avg(file_prefix + "/results/2gram", num_files=30)

	print(len(lstm_conv[0][0]), len(hmm[0][0]))
	assert(len(ours[0][0][1:]) == len(lstm_conv[0][0]) == len(hmm[0][0]) == len(_1g[0][0]) == len(_2g[0][0]) == len(tree_only[0][0][1:]))

	fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

	if legend:
		plt.subplots_adjust(left = 0.05, right = 0.95, bottom=0.23, top=0.98, hspace = 0, wspace = 0)
		fig.set_size_inches(10,4.5)
	else:	
		plt.subplots_adjust(left = 0.05, right = 0.95, bottom=0.12, top=0.96, hspace = 0, wspace = 0)
		fig.set_size_inches(10,4)

	ax1.set_xscale("log", basex = 2)
	ax1.set_xlabel("Number of observations")
	ax1.set_ylabel("Accuracy")
	ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

	L = len(ours[0][0][:-1])
	xrange = range(1, L+1)


	ax1.plot(xrange, ours[0][0][1:], label="DISC (ours)",linestyle=(0, (5,1)), color="b")
	ax1.fill_between(xrange, ours[0][1][1:], ours[0][2][1:], color="b", alpha=0.2)

	ax1.plot(xrange, tree_only[0][0][1:], label="DFA-FT", color="purple")
	ax1.fill_between(xrange, tree_only[0][1][1:], tree_only[0][2][1:], color="purple", alpha=0.2)

	ax1.plot(xrange, lstm_conv[0][0], label="LSTM",  color="r")
	ax1.fill_between(xrange, lstm_conv[0][1], lstm_conv[0][2], color="r", alpha=0.2)

	ax1.plot(xrange, hmm[0][0], label="HMM", color="gray")
	ax1.fill_between(xrange, hmm[0][1], hmm[0][2], color="gray", alpha=0.2)

	ax1.plot(xrange, _1g[0][0], label="One-gram", color="y")
	ax1.fill_between(xrange, _1g[0][1], _1g[0][2], color="y", alpha=0.2)

	ax1.plot(xrange, _2g[0][0], label="Two-gram", color="g")
	ax1.fill_between(xrange, _2g[0][1], _2g[0][2], color="g", alpha=0.2)

	ax1.legend(loc="upper center", bbox_to_anchor = (0.5, -0.15), ncol=3)
	#plt.title(plot_title)
	plt.ylim((0,1))
	#plt.show()

	#plt.clf()
	# print("Ours:", ours[1][0][0], ours[1][0][4])
	# print("Tree only:", tree_only[1][0][0], tree_only[1][0][4])
	# print("LSTM-c:", lstm_conv[1][0][0], lstm_conv[1][0][4])
	# print("HMM:", hmm[1][0][0], hmm[1][0][4])
	# print("1g:", _1g[1][0][0], _1g[1][0][4])
	# print("2g:", _2g[1][0][0], _2g[1][0][4])
	
	print("Ours:", 100*ours[1][0][4], 100*ours[1][1][4])
	print("Tree only:", 100*tree_only[1][0][4], 100*tree_only[1][1][4])
	print("LSTM-c:", 100*lstm_conv[1][0][4], 100*lstm_conv[1][1][4])
	print("HMM:", 100*hmm[1][0][4], 100*hmm[1][1][4])
	print("1g:", 100*_1g[1][0][4], 100*_1g[1][1][4])
	print("2g:", 100*_2g[1][0][4], 100*_2g[1][1][4])


	xmeans = np.arange(5)
	width = 0.1

	def xs(pos):
		return [xmeans[i] + (width)*(pos - 2.5) for i in range(len(xmeans))]

	assert(len(ours[1][0]) == len(lstm_conv[1][0]) == len(hmm[1][0]) == len(_1g[1][0]) == len(_2g[1][0]) == len(tree_only[1][0]))

	ax2.set_xlabel("Percentage of observations")

	plt.sca(ax2)
	plt.xticks(xmeans, ["20%", "40%", "60%", "80%", "100%"])
	ax2.bar(xs(0), ours[1][0], yerr=ours[1][1], width=width, label="DISC (ours)", color="b")
	ax2.bar(xs(1), tree_only[1][0], yerr=tree_only[1][1], width=width, label="DFA-FT", color="purple")
	ax2.bar(xs(2), lstm_conv[1][0], yerr=lstm_conv[1][1], width=width, label="LSTM", color="r")
	ax2.bar(xs(3), hmm[1][0], yerr=hmm[1][1], width=width, label="HMM", color="gray")
	ax2.bar(xs(4), _1g[1][0], yerr=_1g[1][1], width=width, label="One-gram", color="y")
	ax2.bar(xs(5), _2g[1][0], yerr=_2g[1][1], width=width, label="Two-gram", color="g")
	

	ax2.legend(loc="upper center", bbox_to_anchor = (0.5, -0.15), ncol=3)
	plt.ylim((0,1))

	if not legend:
		ax1.get_legend().remove()
		ax2.get_legend().remove()
	plt.show()

# make_plots("traces/crystal", "Crystal Island", legend=False)

# make_plots("traces/starcraft_action_types_only", "Starcraft-a")

# make_plots("traces/alfred", "Alfred", legend=False) 

# make_plots("traces/mit_1", "MIT Activity Recognition")

# make_plots("traces/malware/BOOT_COMPLETED", "BC (Malware)", legend=False)	

# make_plots("traces/malware/BATTERY_LOW", "BL (Malware)")

make_plot_kitchen()

#make_plot_double()

