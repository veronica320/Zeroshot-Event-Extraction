## Make predictions on a dataset with a pretrained EventDetector model. ##
import os
import json
from argparse import ArgumentParser
from Hongming_model_te import EventDetectorTE, hypothesis_dict
from configuration import Config
import torch
from data import IEDataset
from utils import generate_vocabs
from pprint import pprint


def safe_div(num, denom):
	if denom > 0:
		return num / denom
	else:
		return 0


def compute_f1(predicted, gold, matched):
	precision = safe_div(matched, predicted)
	recall = safe_div(matched, gold)
	f1 = safe_div(2 * precision * recall, precision + recall)
	return precision, recall, f1


def convert_arguments(triggers, roles):
	args = set()
	for role in roles:
		trigger_idx = role[0]
		trigger_label = triggers[trigger_idx][-1]
		args.add((trigger_label, role[1], role[2], role[3]))
	return args


def score_graphs(gold_graphs, pred_graphs):
	gold_arg_num = pred_arg_num = arg_idn_num = arg_class_num = 0
	gold_trigger_num = pred_trigger_num = trigger_idn_num = trigger_class_num = 0
	gold_men_num = pred_men_num = men_match_num = 0

	prediction_by_type = dict()
	for tmp_e_type in range(33):
		prediction_by_type[tmp_e_type] = 0

	golden_by_type = dict()
	for tmp_e_type in range(33):
		golden_by_type[tmp_e_type] = 0
	# print(golden_by_type)
	class_correct_by_type = dict()
	for tmp_e_type in range(33):
		class_correct_by_type[tmp_e_type] = 0

	for gold_graph, pred_graph in zip(gold_graphs, pred_graphs):

		# Trigger
		gold_triggers = gold_graph.triggers
		pred_triggers = pred_graph.triggers
		gold_trigger_num += len(gold_triggers)
		pred_trigger_num += len(pred_triggers)
		# print(gold_graphs)
		# print(pred_graphs)
		for trg_start, trg_end, event_type in pred_triggers:
			matched = [item for item in gold_triggers
					   if item[0] == trg_start and item[1] == trg_end]
			if matched:
				trigger_idn_num += 1
				golden_by_type[event_type] += 1


				prediction_by_type[matched[0][-1]] += 1
				if matched[0][-1] == event_type:
					class_correct_by_type[event_type] += 1
					trigger_class_num += 1

		# Argument
		gold_args = convert_arguments(gold_triggers, gold_graph.roles)
		pred_args = convert_arguments(pred_triggers, pred_graph.roles)
		gold_arg_num += len(gold_args)
		pred_arg_num += len(pred_args)
		for pred_arg in pred_args:
			event_type, arg_start, arg_end, role = pred_arg
			gold_idn = {item for item in gold_args
						if item[1] == arg_start and item[2] == arg_end
						and item[0] == event_type}
			if gold_idn:
				arg_idn_num += 1
				gold_class = {item for item in gold_idn if item[-1] == role}
				if gold_class:
					arg_class_num += 1

	trigger_id_prec, trigger_id_rec, trigger_id_f = compute_f1(
		pred_trigger_num, gold_trigger_num, trigger_idn_num)
	trigger_prec, trigger_rec, trigger_f = compute_f1(
		pred_trigger_num, gold_trigger_num, trigger_class_num)
	role_id_prec, role_id_rec, role_id_f = compute_f1(
		pred_arg_num, gold_arg_num, arg_idn_num)
	role_prec, role_rec, role_f = compute_f1(
		pred_arg_num, gold_arg_num, arg_class_num)

	# print('Trigger performance by type')
	# for tmp_e_type in range(33):
	# 	tmp_p, tmp_r, tmp_f1 = compute_f1(prediction_by_type[tmp_e_type], golden_by_type[tmp_e_type], class_correct_by_type[tmp_e_type])
	# 	print(tmp_e_type, 'P:', tmp_p, '| R:', tmp_r, '| F1:', tmp_f1)

	print('Trigger Identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
		trigger_id_prec * 100.0, trigger_id_rec * 100.0, trigger_id_f * 100.0))
	print('Trigger Classification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
		trigger_prec * 100.0, trigger_rec * 100.0, trigger_f * 100.0))
	print('Argument Identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
		role_id_prec * 100.0, role_id_rec * 100.0, role_id_f * 100.0))
	print('Argument Classification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
		role_prec * 100.0, role_rec * 100.0, role_f * 100.0))

	scores = {
		'TC': {'prec': trigger_prec, 'rec': trigger_rec, 'f': trigger_f},
		'TI': {'prec': trigger_id_prec, 'rec': trigger_id_rec,
			   'f': trigger_id_f},
		'AC': {'prec': role_prec, 'rec': role_rec, 'f': role_f},
		'AI': {'prec': role_id_prec, 'rec': role_id_rec, 'f': role_id_f},
	}
	return scores

root_path = ('/shared/lyuqing/probing_for_event')
os.chdir(root_path)

# config
# print(os.getcwd())
# os.chdir('source')
# print(os.getcwd())
config_path = '/shared/hzhangal/Projects/Probing_for_Event/source/config/hm_config.json'
config = Config.from_json_file(config_path)

classification_only = config.classification_only
bert_model_type = config.bert_model_type
srl_args = eval(config.srl_args)
trg_thresh = eval(config.trg_thresh)
arg_thresh = eval(config.arg_thresh)
predicate_type = eval(config.predicate_type)
identify_head = config.identify_head
trg_probe_type = eval(config.trg_probe_type)
pair_premise_strategy = eval(config.pair_premise_strategy)
add_neutral = config.add_neutral
const_premise = eval(config.const_premise)
arg_probe_type = eval(config.arg_probe_type)
tune_on_gdl = eval(config.tune_on_gdl)

frn = config.input_file.split('/')[-1].split('.')[0]

# Model predictions will be written to output_file.
output_file = f"/shared/hzhangal/Projects/Probing_for_Event/output_dir/{frn}_{'cls_' if classification_only else ''}m:{bert_model_type}_t:{trg_thresh}_a:{arg_thresh}_{srl_args}_{predicate_type}_head:{identify_head}_tp:{trg_probe_type}_pps:{pair_premise_strategy}_an:{add_neutral}_cp:{const_premise}_apt:{arg_probe_type}_gdl:{tune_on_gdl}.event.json"

print(f'Model config: {output_file}')
model = EventDetectorTE(config)
model.load_models()

dataset = IEDataset(config.input_file)
vocabs = generate_vocabs([dataset])
dataset.numberize(vocabs)

with open(output_file, 'w') as fw:

	for i, instance in enumerate(dataset):
		print(i, instance.sentence)
		try:
			pred_events = model.predict(instance)
		except IndexError:
			pred_events = list()
		# Gold events and model predictions will also be printed.
		print('Gold events:')
		pprint(instance.events)
		print('Pred events:')
		pprint(pred_events)
		print('\n')

		output = {'doc_id': instance.doc_id,
		          'sent_id': instance.sent_id,
		          'tokens': instance.tokens,
		          'sentence': instance.sentence,
		          'event_mentions': pred_events
		          }

		fw.write(json.dumps(output) + '\n')
		# if i > 20:
		# 	break

gold_dataset = IEDataset(config.input_file)
pred_dataset = IEDataset(output_file)


## Evaluate on all triggers in ACE
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





