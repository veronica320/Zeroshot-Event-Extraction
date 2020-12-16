import os
import json
import allennlp
from allennlp.predictors.predictor import Predictor
from model_te import EventDetectorTE
from model_qa import EventDetectorQA
from configuration import Config
import torch
from data import IEDataset
from utils import generate_vocabs
from pprint import pprint
from scorer import score_graphs


root_path = ('/shared/lyuqing/probing_for_event')
os.chdir(root_path)

# config
config_path = (f'{root_path}/source/config/qa.json')
config = Config.from_json_file(config_path)

frn = config.input_file.split('/')[-1].split('.')[0]

# Model predictions will be written to output_file.
output_file = f"output_dir/QA/TITC_optimal.event.json"

# print(f'Model config: {output_file}')
# model = EventDetectorQA(config)
# model.load_models()
#
#
# dataset = IEDataset(config.input_file)
# vocabs = generate_vocabs([dataset])
# dataset.numberize(vocabs)
#
# with open(output_file, 'w') as fw:
#
# 	for i, instance in enumerate(dataset):
# 		print(i, instance.sentence)
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
#
#
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

