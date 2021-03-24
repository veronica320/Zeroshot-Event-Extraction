## Make predictions on a dataset with a pretrained EventDetector model. ##
import os
from configuration import Config

root_path = ('/shared/lyuqing/probing_for_event')
os.chdir(root_path)

# config
config_path = (f'{root_path}/source/config/te.json')
config = Config.from_json_file(config_path)

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
import json
from argparse import ArgumentParser
from model_te import EventDetectorTE
import torch
from data import IEDataset
from utils import generate_vocabs
from pprint import pprint

classification_only = config.classification_only
gold_trigger = config.gold_trigger

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
srl_model = eval(config.srl_model)

frn = config.input_file.split('/')[-1].split('.')[0]

# Model predictions will be written to output_file.
output_file = f"output_dir/TE/{frn}_{'gt_' if gold_trigger else ''}{'cls_' if classification_only else ''}m:{bert_model_type}_t:{trg_thresh}_a:{arg_thresh}_{srl_args}_{predicate_type}_head:{identify_head}_tp:{trg_probe_type}_pps:{pair_premise_strategy}_an:{add_neutral}_cp:{const_premise}_apt:{arg_probe_type}_gdl:{tune_on_gdl}_srl:{srl_model}.event.json"

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





