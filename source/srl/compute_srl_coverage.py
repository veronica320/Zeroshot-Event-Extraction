import os
import json
from nltk import pos_tag

if __name__ == "__main__":

	root_dir = "/shared/lyuqing/probing_for_event/"
	os.chdir(root_dir)

	input_dir = "data/ACE_oneie/en/event_only"
	input_split = "dev"

	frn_ace = f"{input_dir}/{input_split}.event.json"
	frn_v = 'data/srl_output/illinois_srl/verbSRL_illinois_dev.json'
	frn_n = 'data/srl_output/illinois_srl/nomSRL_illinois_dev.json'

	srl_trg_count = {'verb': 0, 'nom': 0, 'adj': 0, 'others': 0}
	ace_trg_count = {'verb': 0, 'nom': 0, 'adj': 0, 'others': 0}
	single_word_trg_count = 0
	total_trg_count = 0

	with open(frn_v, 'r') as fr_v, open(frn_n, 'r') as fr_n, open(frn_ace, 'r') as fr_ace:
		i = 0

		for line1, line2, line3 in zip(fr_v, fr_n, fr_ace):
			v_srl_output = json.loads(line1)
			n_srl_output = json.loads(line2)
			instance = json.loads(line3)

			tokens = instance['tokens']
			token_poss = pos_tag(tokens)
			poss = [_[1] for _ in token_poss]
			sentence = instance['sentence']

			verbs = [res['verb'] for res in v_srl_output['verbs']]
			noms = [res['nominal'] for res in n_srl_output['nominals']]
			total_srl_preds = verbs + noms
			events = instance['event_mentions']
			for event in events:
				trigger = event['trigger']['text']
				if len(trigger.split()) == 1:
					single_word_trg_count += 1
				total_trg_count += 1

				start, end = event['trigger']['start'], event['trigger']['end']
				if all([pos[:2] == 'VB' for pos in poss[start:end]]):
					ace_trg_count['verb'] += 1
					if trigger in total_srl_preds:
						srl_trg_count['verb'] += 1
					else:
						# print(f"Verb not covered: {event['trigger']}\n{sentence}\n")
						pass
				elif all([pos[:2] == 'NN' for pos in poss[start:end]]):
					ace_trg_count['nom'] += 1
					if trigger in total_srl_preds:
						srl_trg_count['nom'] += 1
					else:
						print(f"Nominal not covered: {event['trigger']}\n{sentence}\n")
				elif all([pos[:2] == 'JJ' for pos in poss[start:end]]):
					ace_trg_count['adj'] += 1
					if trigger in noms or trigger in verbs:
						srl_trg_count['adj'] += 1
					else:
						# print(f"Adj not covered: {event['trigger']}\n{sentence}\n")
						pass
				else:
					ace_trg_count['others'] += 1
					if trigger in noms or trigger in verbs:
						srl_trg_count['adj'] += 1
					else:
						# print(f"Others not covered: {event['trigger']}\n{sentence}\n")
						pass



	print('SRL trigger count:', srl_trg_count, '\nACE trigger count:', ace_trg_count)
	print(sum([value for key,value in srl_trg_count.items()]))
	print(single_word_trg_count, total_trg_count)