import os
from configuration import Config

# repo root dir; change to your own
root_dir = "/shared/lyuqing/Zeroshot-Event-Extraction"
os.chdir(root_dir)

# config
config_path = (f'{root_dir}/source/config/config.json')
config = Config.from_json_file(config_path)

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices

import torch
import json
from model import EventDetector
from data import IEDataset
from utils.datareader import generate_vocabs
from pprint import pprint
from scorer import score_graphs


## Config
# dirs
input_dir = eval(config.input_dir)
split = eval(config.split)
input_file = f"{input_dir}/{split}.event.json"
if "ACE" in input_dir:
	dataset = "ACE"
elif "ERE" in input_dir:
	dataset = "ERE"

# eval settings
setting = eval(config.setting)

# TE-related config
TE_model = config.TE_model.split('/')[-1]
srl_consts = eval(config.srl_consts)
trg_thresh = eval(config.trg_thresh)
trg_probe_type = eval(config.trg_probe_type)

# QA-related config
QA_model = config.QA_model.split('/')[-1]
arg_thresh = eval(config.arg_thresh)
arg_probe_type = eval(config.arg_probe_type)
identify_head = config.identify_head


## Model predictions will be written to output_file.
output_dir = f"output_dir/{dataset}/{split}"
if not os.path.isdir(output_dir):
	os.makedirs(output_dir)
output_file = f"{output_dir}/{split}_{setting}_" \
			  f"TE:{TE_model}_scft:{srl_consts}_tt:{trg_thresh}_tpt:{trg_probe_type}_" \
			  f"QA:{QA_model}_at:{arg_thresh}_apt:{arg_probe_type}_ih:{identify_head}" \
			  f".event.json"

# Predict!
print(f'Model config: {output_file}')

model = EventDetector(config)
model.load_models()

input_dataset = IEDataset(input_file)
vocabs = generate_vocabs([input_dataset])
input_dataset.numberize(vocabs)
with open(output_file, 'w') as fw:
	for i, instance in enumerate(input_dataset):

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
	gold_graphs.append(inst1.graph)
	pred_graphs.append(inst2.graph)

scores = score_graphs(gold_graphs, pred_graphs)
