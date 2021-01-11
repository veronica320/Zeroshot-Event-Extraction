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

YN_QA_model_type = config.YN_QA_model_type # Yes/No QA model
YN_idk = config.YN_idk # IDK class in Yes/No QA model
EX_QA_model_type = config.EX_QA_model_type # Extractive QA model
EX_idk = config.EX_idk # IDK class in Extractive QA model

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

frn = config.input_file.split('/')[-1].split('.')[0]


# Model predictions will be written to output_file.
output_file = f"output_dir/QA/{frn}_{'gt_' if gold_trigger else ''}{'cls_' if classification_only else ''}ynm:{YN_QA_model_type + ('_idk' if YN_idk else '')}_exm:{EX_QA_model_type + ('_idk' if EX_idk else '')}_t:{trg_thresh}_a:{arg_thresh}_{srl_args}_{predicate_type}_head:{identify_head}_pps:{pair_premise_strategy}_an:{add_neutral}_cp:{const_premise}_apt:{arg_probe_type}_gdl:{tune_on_gdl}.event.json"

# Predict
print(f'Model config: {output_file}')
model = EventDetectorQA(config)
model.load_models()

dataset = IEDataset(config.input_file)
vocabs = generate_vocabs([dataset])
dataset.numberize(vocabs)

with open(output_file, 'w') as fw:

	for i, instance in enumerate(dataset):
		print(i, instance.sentence)
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


## Evaluate

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

