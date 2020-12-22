from data import IEDataset
import os
from utils import *

root_path = ('/shared/lyuqing/probing_for_event')
os.chdir(root_path)

arg_name_mapping = {"ATTACK": {"Victim":"Target",
                               "Agent":"Attacker"},
                    "APPEAL": {"Plaintiff":"Prosecutor"}
                    }



input_file = "data/ACE_oneie/en/event_only/dev.event.json"
output_file = "output_dir/QA/dev_gt_ynm:roberta_exm:roberta_t:0.99_a:0.75_all_['verb', 'nom']_head:True_pps:None_an:True_cp:whenNone_apt:ex_gdl:False.event.json"
fwn = "output_dir/QA/extractiveQA_with_goldtrg.txt"
gold_dataset = IEDataset(input_file)
pred_dataset = IEDataset(output_file)

arg_probes_frn = 'source/lexicon/probes/arg_qa_probes_ex.txt'
with open(arg_probes_frn, 'r') as fr:
	arg_probe_lexicon = load_arg_probe_lexicon(fr, 'ex')

## Evaluate on all triggers in ACE
# vocabs = generate_vocabs([gold_dataset, pred_dataset])

# gold_dataset.numberize(vocabs)
# pred_dataset.numberize(vocabs)

gold_graphs, pred_graphs = [], []
with open(fwn, 'w') as fw:
	for inst1, inst2 in zip(gold_dataset, pred_dataset):
		gold_events = inst1['event_mentions']
		pred_events = inst2['event_mentions']

		flag = False
		context = inst1["sentence"]

		for gold_event, pred_event in zip(gold_events, pred_events):

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

			for gold_type in gold_args_by_type:
				gold_arg_name = gold_type + '_Arg'
				gold_arg_texts = gold_args_by_type[gold_type]
				question = arg_probe_lexicon[event_type][gold_arg_name]
				pred_args_same_type = [arg for arg in pred_args if arg['role'] == gold_type]
				pred_args = [arg for arg in pred_args if arg not in pred_args_same_type]

				fw.write(f"\tArgument type: {gold_type}\n")
				fw.write(f"\tQuestion: {question}\n")
				fw.write(f"\t\tGold:\n")
				for gold_arg in gold_arg_texts:
					fw.write(f"\t\t\t{gold_arg}\n")
				fw.write(f"\t\tPredicted:\n")
				if pred_args_same_type:
					for pred_arg in pred_args_same_type:
						pred_arg_text = pred_arg['text']
						pred_arg_conf = pred_arg['confidence']
						fw.write(f"\t\t\t{pred_arg_text} (Confidence: {pred_arg_conf})\n")
				else:
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


			fw.write('\n')



	# pred_events = inst2[]
	# i += 1
	# gold_graphs.append(inst1.graph)
	# pred_graphs.append(inst2.graph)