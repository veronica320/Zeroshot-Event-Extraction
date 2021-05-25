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

root_path = ('/shared/lyuqing/probing_for_event')
os.chdir(root_path)

arg_name_mapping = {"ATTACK": {"Victim":"Target",
                               "Agent":"Attacker"},
                    "APPEAL": {"Plaintiff":"Prosecutor"}
                    }



split = ["dev", "test"][1]
input_file = f"data/ACE_oneie/en/event_only/{split}.event.json"
model_name = ["qamr_roberta-l",
              "qamr-squad2_roberta-l",
              "elior_roberta_squad2",
              "squad2_elior_bert-lc_mnli"
              ][0]
arg_probe_type = ['bool', 'ex', 'ex_wtrg','ex_wtrg+','ex_wtrg_type'][2]

"test_gt_ynm:boolq_roberta_exm:qamr_roberta-l_t:0.99_a:0.0_all_['verb', 'nom']_head:True_pps:None_an:True_cp:whenNone_apt:ex_wtrg_gdl:False_srl:celine_new_allnull_thresh:0.0_cstr:None.event.json"
output_dir = f"output_dir/ACE/{split}/QA"
output_file = f"{output_dir}/{split}_gt_ynm:boolq_roberta_exm:{model_name}_t:0.99_a:0.0_all_['verb', 'nom']_" \
              f"head:True_pps:None_an:True_cp:whenNone_apt:{arg_probe_type}_gdl:False_srl:celine_new_all" \
              f"null_thresh:0.0_cstr:None.event.json"
fwn = f"analysis/{split}_QA_args_wtrg.txt"

gold_dataset = IEDataset(input_file)
pred_dataset = IEDataset(output_file)

arg_probes_frn = 'source/lexicon/probes/ACE/arg_qa_probes_ex_wtrg.txt'
with open(arg_probes_frn, 'r') as fr:
	arg_probe_lexicon = load_arg_probe_lexicon(fr, 'ex')

arg_type_list = []
for event_type in arg_probe_lexicon:
	for arg_type in arg_probe_lexicon[event_type]:
		if arg_type[:-4] not in arg_type_list:
			arg_type_list.append(arg_type[:-4])


gold_graphs, pred_graphs = [], []
with open(fwn, 'w') as fw:
	for inst_id, insts in enumerate(zip(gold_dataset, pred_dataset)):
		inst1, inst2 = insts

		gold_events = inst1['event_mentions']
		pred_events = inst2['event_mentions']

		sent = inst1['sentence']

		for gold_event, pred_event in zip(gold_events, pred_events):

			context = pred_event["arg_textpiece"]

			gold_trigger = gold_event["trigger"]
			pred_trigger = pred_event["trigger"]
			trigger_text = gold_trigger["text"]
			event_type = gold_event["event_type"]
			try:
				assert gold_trigger == pred_trigger
			except AssertionError:
				print(gold_event, pred_event)

			fw.write(f"Sentence: {sent}\n")
			fw.write(f"Context: {context}\n")
			fw.write(f"Trigger: {trigger_text}\n")
			fw.write(f"Event type: {event_type}\n")

			gold_args = gold_event["arguments"]
			pred_args = pred_event["arguments"]

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
				remaining_gold_arg_texts = gold_arg_texts.copy()
				try:
					question = arg_probe_lexicon[event_type][gold_arg_name]
				except KeyError:
					print(event_type)
					print(arg_probe_lexicon[event_type].keys())
				pred_args_same_type = [arg for arg in pred_args if arg['role'] == gold_type]
				pred_args = [arg for arg in pred_args if arg not in pred_args_same_type]

				fw.write(f"\tArgument type: {gold_type}\n")
				fw.write(f"\tQuestion: {question}\n")
				fw.write(f"\t\tGold:\n")
				for gold_arg in gold_arg_texts:
					fw.write(f"\t\t\t{gold_arg}\n")
				fw.write(f"\t\tPredicted:\n")

				gold_type_idx = arg_type_list.index(gold_type)

				if pred_args_same_type:
					for pred_arg in pred_args_same_type:
						pred_arg_text = pred_arg['text']
						pred_arg_conf = pred_arg['confidence']

						answer_cat = None
						answer_label = None
						for gold_arg_text in gold_arg_texts:
							if pred_arg_text == gold_arg_text:
								answer_cat = 'correct'
								remaining_gold_arg_texts.remove(gold_arg_text)
								answer_label = "Correct"
								break
							elif pred_arg_text in gold_arg_text or gold_arg_text in pred_arg_text:
								answer_cat = 'inexact_span'
								remaining_gold_arg_texts.remove(gold_arg_text)
								answer_label = "Inexact span"
								break
						if answer_cat == None:
							answer_cat = 'alt_answer'
							answer_label = "Alternative answer"

						fw.write(f"\t\t\t{pred_arg_text} (Confidence: {pred_arg_conf})\n")
						fw.write(f"\t\t\tAnswer label: {answer_label}\n")

				else:
					fw.write(f"\t\t\tNo Answer\n")
					fw.write(f"\t\t\tAnswer label: \n")

			for pred_arg in pred_args: # remaining pred args
				arg_type = pred_arg['role']
				arg_name = arg_type + '_Arg'
				try:
					question = arg_probe_lexicon[event_type][arg_name]
				except KeyError:
					print(event_type)
					print(arg_probe_lexicon[event_type].keys())
				fw.write(f"\tArgument type: {arg_type}\n")
				fw.write(f"\tQuestion: {question}\n")
				fw.write(f"\t\tGold:\n")
				fw.write(f"\t\t\tNo Answer\n")
				fw.write(f"\t\tPredicted:\n")
				pred_arg_text = pred_arg['text']
				fw.write(f"\t\t\t{pred_arg_text} (Confidence: {pred_arg_conf})\n")
				fw.write(f"\t\t\tAnswer label: \n")

				pred_type_idx = arg_type_list.index(arg_type)

			all_arg_types = set([arg_name[:-4] for arg_name in arg_probe_lexicon[event_type]])
			gold_arg_types = set([gold_arg['role'] for gold_arg in gold_args])
			pred_arg_types = set([pred_arg['role'] for pred_arg in pred_args])

			fw.write('\n')

