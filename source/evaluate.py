import os
import json
from argparse import ArgumentParser
from data import IEDataset
from scorer import score_graphs
from utils import generate_vocabs
from configuration import Config
from nltk import pos_tag
import pprint

root_path = ('/shared/hzhangal/Projects/Probing_for_Event')
os.chdir(root_path)

# config
config_path = 'source/config/hm_config.json'
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

output_file = f"output_dir/{frn}_{'cls_' if classification_only else ''}m:{bert_model_type}_t:{trg_thresh}_a:{arg_thresh}_{srl_args}_{predicate_type}_head:{identify_head}_tp:{trg_probe_type}_pps:{pair_premise_strategy}_an:{add_neutral}_cp:{const_premise}_apt:{arg_probe_type}_gdl:{tune_on_gdl}.event.json"

gold_dataset = IEDataset(config.input_file)
pred_dataset = IEDataset(output_file)

## Evaluate on verb and nominal triggers in ACE only
# for i,inst in enumerate(gold_dataset):
#     tokens = inst['tokens']
#     token_poss = pos_tag(tokens)
#     poss = [_[1] for _ in token_poss]
#     sentence = inst['sentence']
#     filtered_events = []
#     for event in inst['event_mentions']:
#         start, end = event['trigger']['start'], event['trigger']['end']
#         if 'verb' in predicate_type:  # evaluate on events with verb triggers only
#             if all([pos[:2] == 'VB' for pos in poss[start:end]]):
#                 filtered_events.append(event)
#         if 'nom' in predicate_type:
#             if all([pos[:2] == 'NN' for pos in poss[start:end]]):
#                 filtered_events.append(event)
#     inst['event_mentions'] = filtered_events.copy()


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

