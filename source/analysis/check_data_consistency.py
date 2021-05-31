# Check the consistency of ACE data and annotation guideline.
import os
from data import IEDataset
from utils import load_arg_probe_lexicon

root_path = ('/shared/lyuqing/probing_for_event')
os.chdir(root_path)

split = ["dev", "test"][0]
input_file = [f"data/ACE_oneie/en/event_only_cleaned/{split}.event.json", "data/ERE/ERE_oneIE/LDC2015E29/E29.event.json"][1]
gold_dataset = IEDataset(input_file)

if "ACE" in input_file:
	arg_probes_frn = 'source/lexicon/probes/ACE/arg_qa_probes_ex_wtrg.txt'
elif "ERE" in input_file:
	arg_probes_frn = 'source/lexicon/probes/ERE/arg_qa_probes_ex_wtrg.txt'

with open(arg_probes_frn, 'r') as fr:
	arg_probe_lexicon = load_arg_probe_lexicon(fr, 'ex')
gdl_ontology = {}
for event_type in arg_probe_lexicon:
	gdl_ontology[event_type] = []
	for arg_type in arg_probe_lexicon[event_type]:
		gdl_ontology[event_type].append(arg_type[:-4])

# print(gdl_ontology)

dataset_ontology = {}
for inst_id, inst in enumerate(gold_dataset):

	gold_events = inst['event_mentions']
	sent = inst['sentence']

	for gold_event in gold_events:
		gold_trigger = gold_event["trigger"]
		trigger_text = gold_trigger["text"]
		event_type = gold_event["event_type"]

		if event_type not in dataset_ontology:
			dataset_ontology[event_type] = {}

		gold_args = gold_event["arguments"]

		for gold_arg in gold_args:
			gold_arg_text = gold_arg['text']
			gold_arg_type = gold_arg['role']
			if gold_arg_type not in dataset_ontology[event_type]:
				dataset_ontology[event_type][gold_arg_type] = 0
			dataset_ontology[event_type][gold_arg_type] += 1

for event_type in dataset_ontology:
	if event_type not in gdl_ontology:
		print("fEvent type not in guideline ontology: {event_type}")
	for gold_arg_type, count in dataset_ontology[event_type].items():
		if gold_arg_type not in gdl_ontology[event_type]:
			print(event_type, gold_arg_type, count)
			print(dataset_ontology[event_type])
