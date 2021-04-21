from data import IEDataset
import os
from utils import *
import pandas as pd
import numpy as np
import math
from copy import deepcopy
import json
from pprint import pprint
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def is_identical(gold_arg_text, pred_arg_text):
	# Determine if the gold arg and the pred arg is similar enough to be considiered identical.

	n_toks_gold = len(gold_arg_text.split())
	n_toks_pred = len(pred_arg_text.split())

	if (gold_arg_text in pred_arg_text or pred_arg_text in gold_arg_text) \
		and abs(n_toks_pred - n_toks_gold) <= 2:
		return True
	else:
		return False

def add_to_conf_dict(conf, conf_dict):
	if conf < 0.0:
		return -1
	for i in range(len(conf_dict)):
		interval = conf_intervals[i]
		if interval[0] <= conf:
			if interval[1] == 1.0:
				if interval[1] >= conf:
					conf_dict[i] += 1
					return 0
			else:
				if interval[1] > conf:
					conf_dict[i] += 1
					return 0
	return -1



root_path = ('/shared/lyuqing/probing_for_event')
os.chdir(root_path)

arg_name_mapping = {"ATTACK": {"Victim":"Target",
                               "Agent":"Attacker"},
                    "APPEAL": {"Plaintiff":"Prosecutor"}
                    }


input_file = "data/ACE_oneie/en/event_only/dev.event.json"
model_name = ["qamr_roberta-l",
              "qamr-squad2_roberta-l",
              "elior_roberta_squad2",
              "squad2_elior_bert-lc_mnli"
              ][0]
arg_probe_type = ['bool', 'ex', 'ex_wtrg','ex_wtrg+','ex_wtrg_type'][2]

output_file = f"output_dir/QA/dev_gt_ynm:boolq_roberta_exm:{model_name}_t:0.99_a:0.0_all_['verb', 'nom']_" \
              f"head:True_pps:None_an:True_cp:whenNone_apt:{arg_probe_type}_gdl:False_srl:celine_new_all.event.json"
output_path = f"analysis/{model_name}_{arg_probe_type}"
# fwn = f"{output_path}/is_competitive_analysis.json"
comp_frn = f"analysis/qamr_roberta-l/is_competitive_analysis.json"

gold_dataset = IEDataset(input_file)
pred_dataset = IEDataset(output_file)

arg_probes_frn = 'source/lexicon/probes/arg_qa_probes_ex_wtrg.txt'
with open(arg_probes_frn, 'r') as fr:
	arg_probe_lexicon = load_arg_probe_lexicon(fr, 'ex')

arg_type_list = []
for event_type in arg_probe_lexicon:
	for arg_type in arg_probe_lexicon[event_type]:
		if arg_type[:-4] not in arg_type_list:
			arg_type_list.append(arg_type[:-4])


## ranking dictionary
ranks = [i for i in range(11)] + [-1]
ha_rank_counter = np.zeros((len(arg_type_list), len(ranks)), dtype=int)
na_rank_counter = np.zeros((len(arg_type_list), len(ranks)), dtype=int)
cp_rank_counter = np.zeros((len(arg_type_list), len(ranks)), dtype=int)
ncp_rank_counter = np.zeros((len(arg_type_list), len(ranks)), dtype=int)

## confidence dictionary
# (0.0, 0.1), (0.1, 0.2), .... (0.9, 1.0)
conf_intervals = [(round(i*0.1, 1), round(i*0.1+0.1, 1)) for i in range(10)]
ha_conf_counter = np.zeros((len(arg_type_list), len(conf_intervals)), dtype=int)
na_conf_counter = np.zeros((len(arg_type_list), len(conf_intervals)), dtype=int)
cp_conf_counter = np.zeros((len(arg_type_list), len(conf_intervals)), dtype=int)
ncp_conf_counter = np.zeros((len(arg_type_list), len(conf_intervals)), dtype=int)

comp_fr = open(comp_frn, 'r')
comp_fr_insts = json.load(comp_fr)
comp_fr.close()


# annotations for compettive/non-competitive no-answer questions
is_competitive_anno = []

# count of NA at 0th rank when gold answer at 1st rank
count_na_at_0_when_gold_ans_at_1 = 0

gold_graphs, pred_graphs = [], []
for inst_id, insts in enumerate(zip(gold_dataset, pred_dataset, comp_fr_insts)):
	inst1, inst2, comp_inst = insts

	# new_inst = {"doc_id": inst1["doc_id"],
     #            "sent_id": inst1["sent_id"],
	#             "sentence": inst1['sentence'],
     #            "event_mentions":[],
     #            }

	gold_events = inst1['event_mentions']
	pred_events = inst2['event_mentions']
	comp_events = comp_inst["event_mentions"]

	sent = inst1['sentence']

	for gold_event, pred_event, comp_event in zip(gold_events, pred_events, comp_events):

		context = pred_event["arg_textpiece"]
		# print(comp_event.keys())
		na_questions = comp_event["NA quesitons"]

		gold_trigger = gold_event["trigger"]
		pred_trigger = pred_event["trigger"]
		event_type = gold_event["event_type"]
		try:
			assert gold_trigger == pred_trigger
		except:
			print(gold_event, pred_event)



		gold_args = gold_event["arguments"]
		pred_topk_args = pred_event["top_k_arguments"]

		remaining_pred_topk_args = deepcopy(pred_topk_args)

		gold_args_by_type = {}
		for gold_arg in gold_args:
			gold_arg_text = gold_arg['text']
			gold_arg_type = gold_arg['role']

			if event_type in arg_name_mapping:
				if gold_arg_type in arg_name_mapping[event_type]:
					gold_arg_type = arg_name_mapping[event_type][gold_arg_type]

			if gold_arg_type not in gold_args_by_type:
				gold_args_by_type[gold_arg_type] = []
			gold_args_by_type[gold_arg_type].append(gold_arg_text)

		# questions that have gold answers
		for gold_type in gold_args_by_type:
			gold_arg_name = gold_type + '_Arg'
			gold_arg_texts = gold_args_by_type[gold_type]

			pred_topk_args_same_type = pred_topk_args[gold_type]
			if model_name == "qamr-squad2_roberta-l":
				pred_topk_args_same_type = pred_topk_args_same_type[0]
			remaining_pred_topk_args.pop(gold_type)

			gold_type_idx = arg_type_list.index(gold_type)

			# Loop through every gold argument
			for gold_arg_text in gold_arg_texts:
				gold_ans_ranking = -1
				gold_ans_conf = -1

				if pred_topk_args_same_type:
					# Loop through ever predicted arugment
					for rank, pred_arg in enumerate(pred_topk_args_same_type):
						pred_arg_text = pred_arg['answer']
						pred_arg_conf = pred_arg['confidence']
						if not pred_arg_text:
							continue
						if is_identical(gold_arg_text, pred_arg_text):
							gold_ans_ranking = rank
							gold_ans_conf = pred_arg_conf
							break

					gold_ans_rank_idx = ranks.index(gold_ans_ranking)
					if gold_ans_rank_idx == 1:
						if pred_topk_args_same_type[0]["answer"] == None:
							count_na_at_0_when_gold_ans_at_1 += 1
					ha_rank_counter[gold_type_idx][gold_ans_rank_idx] += 1
					add_to_conf_dict(gold_ans_conf, ha_conf_counter[gold_type_idx])

				else:
					ha_rank_counter[gold_type_idx][ranks.index(-1)] += len(gold_arg_texts)


		# no answer questions
		# TODO: shouldn't match by arg types instead of questions
		cp_questions = [_["question"] for _ in na_questions if _["competitive"] == True]
		ncp_questions = [_["question"] for _ in na_questions if _["competitive"] != None and _["competitive"] == False]


		for arg_type, pred_args in remaining_pred_topk_args.items():
			arg_name = arg_type + '_Arg'
			question = arg_probe_lexicon[event_type][arg_name]
			question = question.replace('{trigger}', gold_trigger['text'])

			# new_na_question = {"question":question,
			#                    "competitive": None,
			#                    "top predicted": None,
			#                    "NA in predicted": None,
			#                    }

			# print(f"\tArgument type: {arg_type}\n")
			# print(f"\tQuestion: {question}\n")
			# print(f"\t\tGold: No Answer\n")
			# fw.write(f"\tArgument type: {arg_type}\n")
			# fw.write(f"\tQuestion: {question}\n")
			# fw.write(f"\t\t\tNo Answer\n")

			na_ans_ranking = -1
			na_ans_conf = -1
			if model_name == "qamr-squad2_roberta-l":
				pred_args = pred_args[0]

			for rank, pred_arg in enumerate(pred_args):
				# print(rank)

				# fw.write(f"\t\tPredicted:\n")
				pred_arg_text = pred_arg['answer']
				pred_arg_conf = float(format(pred_arg['confidence'], '.3f'))
				# pred_arg_conf = pred_arg['confidence']


				# if rank == 0:
					# print(f"\t\tTop predicted: {pred_arg_text} (Confidence: {pred_arg_conf})\n")
					# new_na_question["top predicted"] = {"text": pred_arg_text,
					#                                     "confidence":pred_arg_conf
					#                                     }

				if pred_arg_text == None:

					na_ans_ranking = rank
					na_ans_conf = pred_arg_conf
					# new_na_question["NA in predicted"] = {"rank": na_ans_ranking,
					#                                       "confidence": na_ans_conf
					#                                       }

					break
				# fw.write(f"\t\t\t{pred_arg_text} (Confidence: {pred_arg_conf})\n")
#
			pred_type_idx = arg_type_list.index(arg_type)
			na_rank_counter[pred_type_idx][ranks.index(na_ans_ranking)] += 1
			ret = add_to_conf_dict(na_ans_conf, na_conf_counter[pred_type_idx])

			if question in cp_questions:
				cp_rank_counter[pred_type_idx][ranks.index(na_ans_ranking)] += 1
				add_to_conf_dict(na_ans_conf, cp_conf_counter[pred_type_idx])
			elif question in ncp_questions:
				ncp_rank_counter[pred_type_idx][ranks.index(na_ans_ranking)] += 1
				add_to_conf_dict(na_ans_conf, ncp_conf_counter[pred_type_idx])
		# 	new_event["NA quesitons"].append(new_na_question)
		#
		# new_inst["event_mentions"].append(new_event)

	# is_competitive_anno.append(new_inst)

	#
		# all_arg_types = set([arg_name[:-4] for arg_name in arg_probe_lexicon[event_type]])
		# gold_arg_types = set([gold_arg['role'] for gold_arg in gold_args])
		# pred_arg_types = set([pred_arg['role'] for pred_arg in pred_args])
		#
		# correct_na_types = all_arg_types.difference(gold_arg_types.union(pred_arg_types)) # argument types that are correctly answered IDK
		# for type in correct_na_types:
		# 	type_idx = arg_type_list.index(type)
		# 	no_counter[type_idx][no_answer_cats.index('correct')] += 1
		#
		# fw.write('\n')
print(count_na_at_0_when_gold_ans_at_1)

# json.dump(is_competitive_anno, fw, indent=2)
# fw.close()

# print(ha_rank_counter)
# print(na_rank_counter)
# print(ha_conf_counter)
# print(na_conf_counter)

if not os.path.isdir(output_path):
	os.mkdir(output_path)

ha_rank_df = pd.DataFrame(ha_rank_counter, index=arg_type_list, columns=ranks)
ha_rank_df.loc['Total'] = ha_rank_df.sum(axis=0)
ha_rank_df['Total'] = ha_rank_df.sum(axis=1)
ha_rank_df.to_csv(f'{output_path}/ha_rank.csv', index=True, header=True, sep=',')

na_rank_df = pd.DataFrame(na_rank_counter, index=arg_type_list, columns=ranks)
na_rank_df.loc['Total'] = na_rank_df.sum(axis=0)
na_rank_df['Total'] = na_rank_df.sum(axis=1)
na_rank_df.to_csv(f'{output_path}/na_rank.csv', index=True, header=True, sep=',')

ha_conf_df = pd.DataFrame(ha_conf_counter, index=arg_type_list, columns=conf_intervals)
ha_conf_df.loc['Total'] = ha_conf_df.sum(axis=0)
ha_conf_df['Total'] = ha_conf_df.sum(axis=1)
ha_conf_df.to_csv(f'{output_path}/ha_conf.csv', index=True, header=True, sep=',')

na_conf_df = pd.DataFrame(na_conf_counter, index=arg_type_list, columns=conf_intervals)
na_conf_df.loc['Total'] = na_conf_df.sum(axis=0)
na_conf_df['Total'] = na_conf_df.sum(axis=1)
na_conf_df.to_csv(f'{output_path}/na_conf.csv', index=True, header=True, sep=',')

# print(na_conf_df)




cp_rank_df = pd.DataFrame(cp_rank_counter, index=arg_type_list, columns=ranks)
cp_rank_df.loc['Total'] = cp_rank_df.sum(axis=0)
cp_rank_df['Total'] = cp_rank_df.sum(axis=1)
cp_rank_df.to_csv(f'{output_path}/cp_rank.csv', index=True, header=True, sep=',')

ncp_rank_df = pd.DataFrame(ncp_rank_counter, index=arg_type_list, columns=ranks)
ncp_rank_df.loc['Total'] = ncp_rank_df.sum(axis=0)
ncp_rank_df['Total'] = ncp_rank_df.sum(axis=1)
ncp_rank_df.to_csv(f'{output_path}/ncp_rank.csv', index=True, header=True, sep=',')

cp_conf_df = pd.DataFrame(cp_conf_counter, index=arg_type_list, columns=conf_intervals)
cp_conf_df.loc['Total'] = cp_conf_df.sum(axis=0)
cp_conf_df['Total'] = cp_conf_df.sum(axis=1)
cp_conf_df.to_csv(f'{output_path}/cp_conf.csv', index=True, header=True, sep=',')

ncp_conf_df = pd.DataFrame(ncp_conf_counter, index=arg_type_list, columns=conf_intervals)
ncp_conf_df.loc['Total'] = ncp_conf_df.sum(axis=0)
ncp_conf_df['Total'] = ncp_conf_df.sum(axis=1)
ncp_conf_df.to_csv(f'{output_path}/ncp_conf.csv', index=True, header=True, sep=',')

# print(ha_rank_df)
# print(na_rank_df)
# print(ha_conf_df)
# print(na_conf_df)
#


plt.figure()

plt.subplot(2, 2, 1)
total_row = ha_rank_df.loc['Total'].drop(index="Total")
total_row.plot(kind='bar')
plt.xlabel("Ranking of gold answer")
plt.ylabel("Count")
plt.ylim(0,1000)


plt.subplot(2, 2, 2)
total_row = na_rank_df.loc['Total'].drop(index="Total")
total_row.plot(kind='bar')
plt.xlabel("Ranking of NA")
plt.ylim(0,1000)


plt.subplot(2, 2, 3)
total_row = ha_conf_df.loc['Total'].drop(index="Total")
total_row.plot(kind='bar')
plt.xlabel("Confidence of gold answer")
plt.ylabel("Count")
plt.ylim(0,900)

plt.subplot(2, 2, 4)
total_row = na_conf_df.loc['Total'].drop(index="Total")
total_row.plot(kind='bar')
plt.xlabel("Confidence of NA")
plt.ylim(0,900)

plt.suptitle(f'Model: {model_name}')

plt.tight_layout()
plt.savefig(f'{output_path}/NA_HA.png', bbox_inches='tight')





plt.figure()

plt.subplot(2, 2, 1)
total_row = cp_rank_df.loc['Total'].drop(index="Total")
total_row.plot(kind='bar')
plt.xlabel("Competitive: Ranking of NA")
plt.ylabel("Count")
plt.ylim(0,100)


plt.subplot(2, 2, 2)
total_row = ncp_rank_df.loc['Total'].drop(index="Total")
total_row.plot(kind='bar')
plt.xlabel("Non-Competitive: Ranking of NA")
plt.ylim(0,100)


plt.subplot(2, 2, 3)
total_row = cp_conf_df.loc['Total'].drop(index="Total")
total_row.plot(kind='bar')
plt.xlabel("Competitive: Confidence of NA")
plt.ylabel("Count")
plt.ylim(0,100)

plt.subplot(2, 2, 4)
total_row = ncp_conf_df.loc['Total'].drop(index="Total")
total_row.plot(kind='bar')
plt.xlabel("Non-Competitive: Confidence of NA")
plt.ylim(0,100)

plt.suptitle(f'Model: {model_name}')

plt.tight_layout()
plt.savefig(f'{output_path}/comp_noncomp.png', bbox_inches='tight')
