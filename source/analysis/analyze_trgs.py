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

output_file = "output_dir/ACE/TE/dev_cls_m:robertal_t:0.99_a:0.9_['V', 'ARG1']_['verb', 'nom']_head:True_tp:topical_pps:None_an:True_cp:whenNone_apt:manual_gdl:pos_neg_srl:celine_new_all.event.json"
# output_file = "dev_m_robertal_t_0.99_a_0.9_['V', 'ARG1']_['verb', 'nom']_head_True_tp_topical_pps_None_an_True_cp_whenNone_apt_manual_gdl_pos_neg_srl_celine_new_all.event"


gold_dataset = IEDataset(input_file)
pred_dataset = IEDataset(output_file)

trg_probes_frn = 'source/lexicon/probes/ACE/trg_te_probes_topical.txt'
with open(trg_probes_frn, 'r') as fr:
	trg_probe_lexicon = load_trg_probe_lexicon(fr)

trg_type_list = []
for event_type in trg_probe_lexicon:
	trg_type_list.append(event_type)


gold_graphs, pred_graphs = [], []
for inst_id, insts in enumerate(zip(gold_dataset, pred_dataset)):
	inst1, inst2 = insts

	gold_events = inst1['event_mentions']
	pred_events = inst2['event_mentions']

	sent = inst1['sentence']

	remaining_pred_events = deepcopy(pred_events)

	for gold_event in gold_events:
		matched = False
		for pred_event in remaining_pred_events:
			gold_trigger = gold_event["trigger"]
			pred_trigger = pred_event["trigger"]
			if gold_trigger['text'] == pred_trigger['text']:
				if gold_trigger['start'] == pred_trigger['start'] and gold_trigger['end'] == pred_trigger['end']:
					matched = True
					break
				else:
					print(gold_trigger)
					print(pred_trigger)
					label = input("Please manually check if it matches: y/n\n")
					if label == "y":
						matched = True
						break
					else:
						continue
			else:
				continue



		if matched == False:
			print(gold_trigger)
			print(pred_events)
			print(inst1)
			print(inst2)

			raise ValueError("Not matched")

		remaining_pred_events.remove(pred_event)
		context = pred_event["text_piece"]

		gold_event_type = gold_event["event_type"]
		pred_event_type = pred_event["event_type"]

		if gold_event_type != pred_event_type:
			print(f"Sentence: {sent}")
			print(f"Context: {context}")
			print(f"Trigger: {gold_trigger['text']}")
			print(f"Gold: {gold_event_type}")
			print(f"Pred: {pred_event_type}")
			print(f"Reason: ")
			print("\n")
