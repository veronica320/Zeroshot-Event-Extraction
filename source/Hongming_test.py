## Make predictions on a dataset with a pretrained EventDetector model. ##
import os
import json
from argparse import ArgumentParser
from model_te import EventDetectorTE
from configuration import Config
import torch
from data import IEDataset
from utils import generate_vocabs
from pprint import pprint

root_path = ('/shared/hzhangal/Projects/Probing_for_Event/source')
os.chdir(root_path)

# config
# print(os.getcwd())
# os.chdir('source')
# print(os.getcwd())
config_path = 'config/hm_config.json'
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
output_file = f"output_dir/{frn}_{'cls_' if classification_only else ''}m:{bert_model_type}_t:{trg_thresh}_a:{arg_thresh}_{srl_args}_{predicate_type}_head:{identify_head}_tp:{trg_probe_type}_pps:{pair_premise_strategy}_an:{add_neutral}_cp:{const_premise}_apt:{arg_probe_type}_gdl:{tune_on_gdl}.event.json"

print(f'Model config: {output_file}')
model = EventDetectorTE(config)
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





