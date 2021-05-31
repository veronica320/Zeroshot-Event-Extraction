import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import csv
import os
import matplotlib.patches as mpatches


plt.rcParams["font.family"] = "Times New Roman"

def load_count_f(frn):
	count_list = []
	with open(frn, 'r') as fr:
		reader = csv.reader(fr)
		for i, row in enumerate(reader):
			if i == 0:
				continue
			count_list.append({"type":row[0], "count":row[1], "class":row[2]})
	return count_list


def plot_histogram(count_list, out_fn):
	if "trg" in out_fn:
		fig = plt.figure(figsize=(4.5, 3.7))
		# ax = fig.add_subplot(121)
	if "arg" in out_fn:
		fig = plt.figure(figsize=(5, 4))
		# ax = fig.add_subplot(122)

	patterns = {"Model": "-",
	            "Usage": "\\",
	            "Task": "/",
	            "Other": ""}

	cmap = plt.cm.get_cmap('BuPu')

	colors = {"Model": cmap(0.2),
	          "Usage": cmap(0.5),
	          "Task": cmap(0.75),
	          "Other": cmap(0.99)}

	ax = fig.add_subplot(111)

	for i, item in enumerate(count_list):
		type = item["type"]
		count = int(item["count"])
		fault_class = item["class"]
		color = colors[fault_class]
		hatch = patterns[fault_class]
		ax.bar(type, count, color=color, edgecolor='black', alpha=0.4, hatch=hatch, width=0.7)

	plt.xticks(rotation=45, ha='right')
	plt.ylim(0, 30)
	if "trg" in out_fn:
		plt.ylabel("Count")


	circs = []
	for fault_class in colors:
		circ = mpatches.Patch(facecolor=colors[fault_class], alpha=0.4, hatch=patterns[fault_class], label=fault_class)
		circs.append(circ)

	ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", ncol=4, handles=circs)

	plt.tight_layout()

	plt.savefig(out_fn, bbox_inches='tight', dpi=1200)


if __name__ == "__main__":
	root_path = ('/shared/lyuqing/probing_for_event')
	os.chdir(root_path)

	output_path = f"analysis/error_plots/"
	if not os.path.isdir(output_path):
		os.mkdir(output_path)

	trg_count_frn = f"{output_path}/trg_error_count.csv"
	trg_count_list = load_count_f(trg_count_frn)
	trg_out_fn = f'{output_path}/trg_error_type.png'


	arg_count_frn = f"{output_path}/arg_error_count.csv"
	arg_count_list = load_count_f(arg_count_frn)
	arg_out_fn = f'{output_path}/arg_error_type.png'

	plot_histogram(trg_count_list, trg_out_fn)
	plot_histogram(arg_count_list, arg_out_fn)






