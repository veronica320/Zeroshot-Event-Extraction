from data import IEDataset
import os
from utils import *
import pandas as pd
import numpy as np

root_path = ('/shared/lyuqing/probing_for_event')
os.chdir(root_path)

arg_name_mapping = {"ATTACK": {"Victim":"Target",
                               "Agent":"Attacker"},
                    "APPEAL": {"Plaintiff":"Prosecutor"}
                    }



input_file = "data/ACE_oneie/en/event_only/dev.event.json"
output_file = "output_dir/QA/dev_gt_ynm:boolq_roberta_exm:qamr_roberta-l_t:0.99_a:0.0_all_['verb', 'nom']_head:True_pps:None_an:True_cp:whenNone_apt:ex_wtrg_gdl:False.event.json"
fwn = "output_dir/QA/dev_gt_ynm:boolq_roberta_exm:qamr_roberta-l_t:0.99_a:0.0_all_['verb', 'nom']_head:True_pps:None_an:True_cp:whenNone_apt:ex_wtrg_gdl:False.txt"
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

has_answer_cats = ['correct', 'inexact_span', 'alt_answer', 'no_answer']
na_idx = has_answer_cats.index('no_answer')
no_answer_cats = ['correct', 'alt_answer']


has_counter = np.zeros((len(arg_type_list), len(has_answer_cats))) # has answer
no_counter = np.zeros((len(arg_type_list), len(no_answer_cats))) # no answer

gold_graphs, pred_graphs = [], []
with open(fwn, 'w') as fw:
	for inst1, inst2 in zip(gold_dataset, pred_dataset):
		gold_events = inst1['event_mentions']
		pred_events = inst2['event_mentions']

		flag = False
		sent = inst1['sentence']
		for gold_event, pred_event in zip(gold_events, pred_events):

			context = pred_event["arg_textpiece"]
			fw.write(f"Sentence: {sent}\n")
			fw.write(f"Context: {context}\n")

			gold_trigger = gold_event["trigger"]
			pred_trigger = pred_event["trigger"]
			event_type = gold_event["event_type"]
			assert gold_trigger == pred_trigger

			fw.write(f"Event type: {event_type}\n")
			fw.write(f"Trigger: {gold_trigger['text']}\n")
			fw.write("Arguments:\n")

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

			for gold_type in gold_args_by_type: # questions that have gold answers
				gold_arg_name = gold_type + '_Arg'
				gold_arg_texts = gold_args_by_type[gold_type]
				remaining_gold_arg_texts = gold_arg_texts.copy()
				question = arg_probe_lexicon[event_type][gold_arg_name]
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
						for gold_arg_text in gold_arg_texts:
							if pred_arg_text == gold_arg_text:
								answer_cat = 'correct'
								remaining_gold_arg_texts.remove(gold_arg_text)
								break
							elif pred_arg_text in gold_arg_text or gold_arg_text in pred_arg_text:
								answer_cat = 'inexact_span'
								remaining_gold_arg_texts.remove(gold_arg_text)
								break
						if answer_cat == None:
							answer_cat = 'alt_answer'

						answer_cat_idx = has_answer_cats.index(answer_cat)
						has_counter[gold_type_idx][answer_cat_idx] += 1
						fw.write(f"\t\t\t{pred_arg_text} (Confidence: {pred_arg_conf}) ({answer_cat})\n")
						has_counter[gold_type_idx][na_idx] += len(remaining_gold_arg_texts)
				else:
					has_counter[gold_type_idx][na_idx] += len(gold_arg_texts)
					fw.write(f"\t\t\tNo Answer\n")

			for pred_arg in pred_args: # remaining pred args
				arg_type = pred_arg['role']
				arg_name = arg_type + '_Arg'
				question = arg_probe_lexicon[event_type][arg_name]
				fw.write(f"\tArgument type: {arg_type}\n")
				fw.write(f"\tQuestion: {question}\n")
				fw.write(f"\t\tGold:\n")
				fw.write(f"\t\t\tNo Answer\n")
				fw.write(f"\t\tPredicted:\n")
				pred_arg_text = pred_arg['text']
				pred_arg_conf = pred_arg['confidence']
				fw.write(f"\t\t\t{pred_arg_text} (Confidence: {pred_arg_conf})\n")

				pred_type_idx = arg_type_list.index(arg_type)
				no_counter[pred_type_idx][no_answer_cats.index('alt_answer')] += 1

			all_arg_types = set([arg_name[:-4] for arg_name in arg_probe_lexicon[event_type]])
			gold_arg_types = set([gold_arg['role'] for gold_arg in gold_args])
			pred_arg_types = set([pred_arg['role'] for pred_arg in pred_args])

			correct_na_types = all_arg_types.difference(gold_arg_types.union(pred_arg_types)) # argument types that are correctly answered IDK
			for type in correct_na_types:
				type_idx = arg_type_list.index(type)
				no_counter[type_idx][no_answer_cats.index('correct')] += 1

			fw.write('\n')



# print(has_counter)
# print(no_counter)

has_answer_df = pd.DataFrame(has_counter, index=arg_type_list, columns=has_answer_cats)
has_answer_df.loc['Total'] = has_answer_df.sum(axis=0)
has_answer_df['Total'] = has_answer_df.sum(axis=1)
has_answer_df.to_csv('has_answer_a:0.0.csv', index=True, header=True, sep=',')

no_answer_df = pd.DataFrame(no_counter, index=arg_type_list, columns=no_answer_cats)
no_answer_df.loc['Total'] = no_answer_df.sum(axis=0)
no_answer_df['Total'] = no_answer_df.sum(axis=1)
no_answer_df.to_csv('no_answer_a:0.0.csv', index=True, header=True, sep=',')
