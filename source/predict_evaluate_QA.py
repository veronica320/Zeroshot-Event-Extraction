import os
from configuration import Config

root_path = ('/shared/lyuqing/probing_for_event')
os.chdir(root_path)

# config
config_path = (f'{root_path}/source/config/qa.json')
config = Config.from_json_file(config_path)

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices

import torch
import json
import allennlp
from allennlp.predictors.predictor import Predictor
from model_te import EventDetectorTE
from model_qa import EventDetectorQA
from data import IEDataset
from utils import generate_vocabs
from pprint import pprint
from scorer import score_graphs

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

input_path = eval(config.input_path)
split = eval(config.split)

trigger_input_mode = ["scratch", "gold_ti", "gold_titc"][1]

# if trigger_input_mode == "scratch":
# 	# input_file = "output_dir/ACE/test/TE/test_m:robertal_t:0.99_a:0.9_['V', 'ARG1']_['verb', 'nom']_head:True_tp:topical_pps:None_an:True_cp:whenNone_apt:manual_gdl:pos_neg_srl:celine_new_all.event.json"
# 	input_file = "output_dir/ERE/E29/TE/E29_m:robertal_t:0.99_a:0.9_['V', 'ARG1']_['verb', 'nom']_head:True_tp:topical_pps:None_an:True_cp:whenNone_apt:manual_gdl:pos_neg_srl:celine_new_all.event.json"
# elif trigger_input_mode == "gold_ti":
# 	# input_file = "output_dir/ACE/test/TE/test_cls_m:robertal_t:0.99_a:0.9_['V', 'ARG1']_['verb', 'nom']_head:True_tp:topical_pps:None_an:True_cp:whenNone_apt:manual_gdl:pos_neg_srl:celine_new_all.event.json"
# 	input_file = "output_dir/ERE/E29/TE/E29_cls_m:robertal_t:0.99_a:0.9_['V', 'ARG1']_['verb', 'nom']_head:True_tp:topical_pps:None_an:True_cp:whenNone_apt:manual_gdl:pos_neg_srl:celine_new_all.event.json"
# elif trigger_input_mode == "gold_titc":
input_file = f"{input_path}/{split}.event.json"


if "ACE" in input_file:
	output_dir = f"output_dir/ACE/{split}/QA"
elif "ERE" in input_file:
	output_dir = f"output_dir/ERE/{split}"


# Model predictions will be written to output_file.
if trigger_input_mode == "scratch":
	output_file = f"{output_dir}/{split}_best_trigger_scratch_{'gt_' if gold_trigger else ''}{'cls_' if classification_only else ''}ynm:{YN_QA_model_name}_exm:{EX_QA_model_name}_t:{trg_thresh}_a:{arg_thresh}_{srl_args}_{predicate_type}_head:{identify_head}_pps:{pair_premise_strategy}_an:{add_neutral}_cp:{const_premise}_apt:{arg_probe_type}_gdl:{tune_on_gdl}.event.json"
elif trigger_input_mode == "gold_ti":
	output_file = f"{output_dir}/{split}_best_trigger_goldti_{'gt_' if gold_trigger else ''}{'cls_' if classification_only else ''}ynm:{YN_QA_model_name}_exm:{EX_QA_model_name}_t:{trg_thresh}_a:{arg_thresh}_{srl_args}_{predicate_type}_head:{identify_head}_pps:{pair_premise_strategy}_an:{add_neutral}_cp:{const_premise}_apt:{arg_probe_type}_gdl:{tune_on_gdl}.event.json"
elif trigger_input_mode == "gold_titc":
	output_file = f"{output_dir}/{split}_{'gt_' if gold_trigger else ''}{'cls_' if classification_only else ''}" \
	              f"ynm:{YN_QA_model_name}_exm:{EX_QA_model_name}_t:{trg_thresh}_a:{arg_thresh}_{srl_args}_" \
	              f"{predicate_type}_head:{identify_head}_pps:{pair_premise_strategy}_an:{add_neutral}_" \
	              f"cp:{const_premise}_apt:{arg_probe_type}_gdl:{tune_on_gdl}_srl:{srl_model}" \
	              f"null_thresh:{null_score_diff_threshold}_cstr:{global_constraint}" \
	              f".event.json"

# ## Predict
# print(f'Model config: {output_file}')
# model = EventDetectorQA(config)
# model.load_models()
#
# dataset = IEDataset(input_file)
# vocabs = generate_vocabs([dataset])
# dataset.numberize(vocabs)
#
# with open(output_file, 'a') as fw:
#
# 	for i, instance in enumerate(dataset):
# 		print(i, instance.sentence)
# 		# if i > 100:
# 		# 	break
# 		pred_events = model.predict(instance)
#
# 		# Gold events and model predictions will also be printed.
# 		print('Gold events:')
# 		pprint(instance.events)
# 		print('Pred events:')
# 		pprint(pred_events)
# 		print('\n')
#
# 		output = {'doc_id': instance.doc_id,
# 		          'sent_id': instance.sent_id,
# 		          'tokens': instance.tokens,
# 		          'sentence': instance.sentence,
# 		          'event_mentions': pred_events
# 		          }
#
# 		fw.write(json.dumps(output) + '\n')

## Evaluate

gold_dataset = IEDataset(input_file)
pred_dataset = IEDataset(output_file)

vocabs = generate_vocabs([gold_dataset, pred_dataset])

gold_dataset.numberize(vocabs)
pred_dataset.numberize(vocabs)

gold_graphs, pred_graphs = [], []

i = 0
for inst1, inst2 in zip(gold_dataset, pred_dataset):
	i += 1
	gold_graphs.append(inst1.graph)
	pred_graphs.append(inst2.graph)

scores = score_graphs(gold_graphs, pred_graphs)
print(scores)

