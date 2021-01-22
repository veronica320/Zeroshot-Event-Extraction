import os
from configuration import Config

root_path = ('/shared/lyuqing/probing_for_event')
os.chdir(root_path)

# config
config_path = (f'{root_path}/source/config/mte.json')
config = Config.from_json_file(config_path)

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices

import json
from argparse import ArgumentParser
from model_mte import MultilingEventDetectorTE
import torch
from data import IEDataset
from utils import generate_vocabs
from pprint import pprint
from scorer import score_graphs


classification_only = config.classification_only
gold_trigger = config.gold_trigger

bert_model_type = config.bert_model_type
srl_args = eval(config.srl_args)
trg_thresh = eval(config.trg_thresh)
predicate_type = eval(config.predicate_type)
trg_probe_type = eval(config.trg_probe_type)
pair_premise_strategy = eval(config.pair_premise_strategy)
add_neutral = config.add_neutral
const_premise = eval(config.const_premise)

frn = config.input_file.split('/')[-1].split('.')[0]

# Model predictions will be written to output_file.
output_file = f"output_dir/mTE/{frn}_{'gt_' if gold_trigger else ''}{'cls_' if classification_only else ''}m:{bert_model_type}_t:{trg_thresh}_{srl_args}_{predicate_type}_tp:{trg_probe_type}_pps:{pair_premise_strategy}_an:{add_neutral}_cp:{const_premise}.event.json"

print(f'Model config: {output_file}')
model = MultilingEventDetectorTE(config)
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



