import os
from configuration import Config

root_path = ('/shared/lyuqing/probing_for_event')
os.chdir(root_path)

# config
config_path = (f'{root_path}/source/config/te.json')
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

evaluate_only = config.evaluate_only

bert_model_type = config.bert_model_type

classification_only = config.classification_only
gold_trigger = config.gold_trigger

srl_args = eval(config.srl_args)
trg_thresh = eval(config.trg_thresh)
arg_thresh = eval(config.arg_thresh)
predicate_type = eval(config.predicate_type)
add_neutral = config.add_neutral
identify_head = config.identify_head
trg_probe_type = eval(config.trg_probe_type)
tune_on_gdl = eval(config.tune_on_gdl)
const_premise = eval(config.const_premise)
pair_premise_strategy = eval(config.pair_premise_strategy)
arg_probe_type = eval(config.arg_probe_type)
srl_model = eval(config.srl_model)

input_path = eval(config.input_path)
split = eval(config.split)

input_file = f"{input_path}/{split}.event.json"

if "ACE" in input_path:
	output_dir = f"output_dir/ACE/{split}/TE"
elif "ERE" in input_path:
	output_dir = f"output_dir/ERE/{split}/TE"

# Model predictions will be written to output_file.
output_file = f"{output_dir}/{split}_{'gt_' if gold_trigger else ''}{'cls_' if classification_only else ''}" \
              f"m:{bert_model_type}_t:{trg_thresh}_a:{arg_thresh}_{srl_args}_" \
              f"{predicate_type}_head:{identify_head}_tp:{trg_probe_type}_pps:{pair_premise_strategy}_an:{add_neutral}_" \
              f"cp:{const_premise}_apt:{arg_probe_type}_gdl:{tune_on_gdl}_srl:{srl_model}" \
              f".event.json"


# Predict
print(f'Model config: {output_file}')

if not evaluate_only:
	model = EventDetectorTE(config)
	model.load_models()

	dataset = IEDataset(input_file)
	vocabs = generate_vocabs([dataset])
	dataset.numberize(vocabs)

	with open(output_file, 'a') as fw:

		for i, instance in enumerate(dataset):
			print(i, instance.sentence)
			# if i > 100:
			# 	break
			# if i < 412:
			# 	continue
			pred_events = model.predict(instance)

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
			fw.flush()

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
	if inst1.doc_id != inst2.doc_id or inst1.sent_id != inst2.sent_id:
		print(i)
		print(inst1.sentence)
		break
	gold_graphs.append(inst1.graph)
	pred_graphs.append(inst2.graph)

scores = score_graphs(gold_graphs, pred_graphs)
print(scores)

