from data import IEDataset

import os

root_path = ('/shared/lyuqing/probing_for_event')
os.chdir(root_path)

orig_input_file = f"data/ACE_oneie/en/event_only/test.event.json"
annotated_file = f"test_QA_args_wtrg_annotated.txt"

target_error_type = ["grm","context","broad"][0]


arg_name_mapping = {"ATTACK": {"Victim":"Target",
                               "Agent":"Attacker"},
                    "APPEAL": {"Plaintiff":"Prosecutor"}
                    }
gold_dataset = IEDataset(orig_input_file)

inst_ids_of_target_error_type = []
with open(annotated_file, 'r') as annotated_f:
	inst_id = -1
	sentence, arg_type  = None, None
	for line in annotated_f:
		line = line.strip()
		if line.startswith("Sentence"):
			sentence = line.split(": ")[1]
			inst_id += 1
		if line.startswith("Argument type"):
			arg_type = line.split(": ")[1]
		if line.startswith("Answer label"):
			answer_labels = line.split(":")[1].strip().lower()
			answer_labels = answer_labels.split(",")
			answer_labels = [label.strip() for label in answer_labels]
			if target_error_type in answer_labels:
				inst_ids_of_target_error_type.append({"id": inst_id, "sentence": sentence})


for inst_id, inst in enumerate(zip(gold_dataset)):
	gold_events = inst['event_mentions']

	sent = inst['sentence']

	for gold_event in gold_events:

		context = gold_event["arg_textpiece"]

		gold_trigger = gold_event["trigger"]
		trigger_text = gold_trigger["text"]
		event_type = gold_event["event_type"]
		gold_args = gold_event["arguments"]

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

		for pred_arg in pred_args:  # remaining pred args
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

