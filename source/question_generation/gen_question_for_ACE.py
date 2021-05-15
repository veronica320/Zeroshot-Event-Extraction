import sys
sys.path.append('../')

import os
from configuration import Config
cuda = 7
os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
from model import QuestionGenerationModel
import json
from data import IEDataset
from utils import generate_vocabs
from pprint import pprint
import re

root_path = "/shared/lyuqing/probing_for_event"
os.chdir(root_path)

model_path = "output_model_dir/question_generation_model/model.tar.gz"
model = QuestionGenerationModel(model_path, cuda_device = 0)

# config
config_path = (f'{root_path}/source/config/qa.json')
config = Config.from_json_file(config_path)

n_multiple_occurrences = 0

def gen_question(context, answer):
	global n_multiple_occurrences
	answer_text = answer["text"].lower()
	context = context.lower()

	char_spans = []
	# TODO: change to span-based matching
	for m in re.finditer(answer_text, context):
		char_spans.append((m.start(), m.end()))

	if len(char_spans) == 0:
		raise ValueError(f"Answer is not found in context: {answer_text}, {context}\n")

	elif len(char_spans) > 1:
		print("Multiple occurrences of answer found in context.\n")
		n_multiple_occurrences += 1

	char_span = char_spans[0]
	start, end = char_span
	generated_question = model.generate(context, start, end)

	return generated_question


YN_QA_model_name = config.YN_QA_model_name # Yes/No QA model
EX_QA_model_name = config.EX_QA_model_name # Extractive QA model

classification_only = config.classification_only
gold_trigger = config.gold_trigger

srl_args = eval(config.srl_args)
trg_thresh = eval(config.trg_thresh)
arg_thresh = eval(config.arg_thresh)
predicate_type = eval(config.predicate_type)
add_neutral = config.add_neutral
identify_head = config.identify_head
tune_on_gdl = eval(config.tune_on_gdl)
const_premise = eval(config.const_premise)
pair_premise_strategy = eval(config.pair_premise_strategy)
arg_probe_type = eval(config.arg_probe_type)
srl_model = eval(config.srl_model)
null_score_diff_threshold = config.null_score_diff_threshold
global_constraint = eval(config.global_constraint)

frn = config.input_file.split('/')[-1].split('.')[0]
output_file = f"output_dir/QA/{frn}_{'gt_' if gold_trigger else ''}{'cls_' if classification_only else ''}" \
              f"ynm:{YN_QA_model_name}_exm:{EX_QA_model_name}_t:{trg_thresh}_a:{arg_thresh}_{srl_args}_" \
              f"{predicate_type}_head:{identify_head}_pps:{pair_premise_strategy}_an:{add_neutral}_" \
              f"cp:{const_premise}_apt:{arg_probe_type}_gdl:{tune_on_gdl}_srl:{srl_model}" \
              f"null_thresh:{null_score_diff_threshold}_cstr:{global_constraint}" \
              f".event.json"

gold_dataset = IEDataset(config.input_file)
pred_dataset = IEDataset(output_file)

cor_arg_dict = {'trigger in Q': 0, 'trigger not in Q': 0}
inc_arg_dict = {'trigger in Q': 0, 'trigger not in Q': 0}

for inst_id, insts in enumerate(zip(gold_dataset, pred_dataset)):
	# if inst_id > 10:
	# 	break

	inst1, inst2 = insts

	gold_events = inst1['event_mentions']
	pred_events = inst2['event_mentions']

	tokens = inst1["tokens"]

	sent = inst1['sentence']

	for gold_event, pred_event in zip(gold_events, pred_events):

		context = pred_event["arg_textpiece"]

		gold_trigger = gold_event["trigger"]
		pred_trigger = pred_event["trigger"]
		trigger_text = gold_trigger["text"]

		event_type = gold_event["event_type"]
		try:
			assert gold_trigger == pred_trigger
		except:
			print(gold_event, pred_event)
			continue

		gold_args = gold_event["arguments"]
		pred_args = pred_event["arguments"]

		for pred_arg in pred_args:
			generated_question = gen_question(context, pred_arg)
			
			pred_span = pred_arg["start"], pred_arg["end"]
			pred_role = pred_arg["role"]
			
			is_correct = False
			for gold_arg in gold_args:
				gold_span = gold_arg["start"], gold_arg["end"]
				gold_role = gold_arg["role"]
				if pred_span == gold_span and pred_role == gold_role:
					is_correct = True


			if is_correct:	
				if trigger_text in generated_question:
					cor_arg_dict['trigger in Q'] += 1
				else:
					cor_arg_dict['trigger not in Q'] += 1

			else:
				if trigger_text in generated_question:
					inc_arg_dict['trigger in Q'] += 1
				else:
					inc_arg_dict['trigger not in Q'] += 1

print(n_multiple_occurrences)
print(cor_arg_dict)
print(inc_arg_dict)