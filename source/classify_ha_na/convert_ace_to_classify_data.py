# Converts ACE QA data to classification data.
import os
import json
import csv
from data import IEDataset
from utils import load_arg_probe_lexicon
from copy import deepcopy


arg_name_mapping = {"ATTACK": {"Victim":"Target",
                               "Agent":"Attacker"},
                    "APPEAL": {"Plaintiff":"Prosecutor"}
                    }

root_path = ('/shared/lyuqing/probing_for_event')
os.chdir(root_path)

input_file = "data/ACE_oneie/en/event_only/dev.event.json"
gold_dataset = IEDataset(input_file)

arg_probe_type = ['bool', 'ex', 'ex_wtrg','ex_wtrg+','ex_wtrg_type'][1]
arg_probes_frn = f'source/lexicon/probes/arg_qa_probes_{arg_probe_type}.txt'
with open(arg_probes_frn, 'r') as fr:
	arg_probe_lexicon = load_arg_probe_lexicon(fr, 'ex')
arg_type_list = []
for event_type in arg_probe_lexicon:
	for arg_type in arg_probe_lexicon[event_type]:
		if arg_type[:-4] not in arg_type_list:
			arg_type_list.append(arg_type[:-4])

model_name = ["qamr_roberta-l",
              "qamr-squad2_roberta-l",
              "elior_roberta_squad2",
              "squad2_elior_bert-lc_mnli"
              ][0]
output_file = f"output_dir/QA/dev_gt_ynm:boolq_roberta_exm:{model_name}_t:0.99_a:0.0_all_['verb', 'nom']_" \
              f"head:True_pps:None_an:True_cp:whenNone_apt:{arg_probe_type}_gdl:False_srl:celine_new_all.event.json"
pred_dataset = IEDataset(output_file)


new_dir = f"data/ACE_ha_na_cls/{arg_probe_type}"
if not os.path.isdir(new_dir):
	os.mkdir(new_dir)

fwn = f"{new_dir}/dev.tsv"
fw = open(fwn, 'w')
writer = csv.writer(fw, delimiter="\t")
writer.writerow(["idx", "sentence1", "sentence2", "label"])

# number of ha vs. na classification examples
cls_example_id = 0

for inst_id, insts in enumerate(zip(gold_dataset, pred_dataset)):

	inst1, inst2 = insts

	sent = inst1['sentence']
	gold_events = inst1['event_mentions']
	pred_events = inst2['event_mentions']

	for gold_event, pred_event in zip(gold_events, pred_events):

		context = pred_event["arg_textpiece"]

		gold_trigger = gold_event["trigger"]
		pred_trigger = pred_event["trigger"]
		event_type = gold_event["event_type"]
		try:
			assert gold_trigger == pred_trigger
		except:
			print(gold_event, pred_event)
			continue

		gold_args = gold_event["arguments"]
		pred_topk_args = pred_event["top_k_arguments"]

		# remaining_pred_topk_args = deepcopy(pred_topk_args)

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


		# has-answer questions
		for gold_type in gold_args_by_type:
			gold_arg_name = gold_type + '_Arg'
			gold_arg_texts = gold_args_by_type[gold_type]

			gold_type_idx = arg_type_list.index(gold_type)
			# remaining_pred_topk_args.pop(gold_type)

			question = arg_probe_lexicon[event_type][gold_arg_name]
			question = question.replace('{trigger}', gold_trigger['text'])
			question += '?'

			row = [cls_example_id, context, question, 2]
			writer.writerow(row)
			cls_example_id += 1

		# no-answer questions
		ha_arg_types = set(gold_args_by_type.keys())
		all_arg_types_in_event = set([arg_name[:-4] for arg_name in arg_probe_lexicon[event_type]])
		na_arg_types = all_arg_types_in_event - ha_arg_types

		for arg_type in na_arg_types:
			arg_name = arg_type + '_Arg'
			question = arg_probe_lexicon[event_type][arg_name]
			question = question.replace('{trigger}', gold_trigger['text'])
			question += '?'

			row = [cls_example_id, context, question, 0]
			writer.writerow(row)
			cls_example_id += 1

fw.close()

