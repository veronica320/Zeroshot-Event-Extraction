import os
from configuration import Config

root_path = ('/shared/lyuqing/probing_for_event')
os.chdir(root_path)

# config
config_path = (f'{root_path}/source/config/mqa.json')
config = Config.from_json_file(config_path)

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices

import json
from argparse import ArgumentParser
from model_mqa import MultilingEventDetectorQA
import torch
from data import IEDataset
from utils import generate_vocabs
from pprint import pprint
from scorer import score_graphs


classification_only = config.classification_only
gold_trigger = config.gold_trigger

EX_QA_model_name = config.EX_QA_model_name
add_neutral = config.add_neutral
arg_thresh = eval(config.arg_thresh)
arg_probe_type = eval(config.arg_probe_type)

frn = config.input_file.split('/')[-1].split('.')[0]

# Model predictions will be written to output_file.
output_file = f"output_dir/mQA/{frn}_{'gt_' if gold_trigger else ''}{'cls_' if classification_only else ''}_exm:{EX_QA_model_name}_a:{arg_thresh}_an:{add_neutral}_apt:{arg_probe_type}.event.json"

print(f'Model config: {output_file}')
model = MultilingEventDetectorQA(config)
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



