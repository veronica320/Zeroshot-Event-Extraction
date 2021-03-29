import os
import json
from nltk import pos_tag
from pprint import pprint

if __name__ == "__main__":

	root_dir = "/shared/lyuqing/probing_for_event/"
	os.chdir(root_dir)

	input_dir = "data/ACE_oneie/en/event_only"
	input_split = "dev"

	frn_ace = f"{input_dir}/{input_split}.event.json"

	srl_model_name = ["celine_old", "celine_new", "celine_new_all", "illinois"][3]

	frn_v = f'data/srl_output/{srl_model_name}/verbSRL_{srl_model_name}_dev.json'
	frn_n = f'data/srl_output/{srl_model_name}/nomSRL_{srl_model_name}_dev.json'

	srl_trg_count = {'verb': 0, 'nom': 0, 'adj': 0, 'others': 0}
	ace_trg_count = {'verb': 0, 'nom': 0, 'adj': 0, 'others': 0}
	single_word_trg_count = 0
	total_trg_count = 0

	uncovered_dict = {'verb': {}, 'nom': {}, 'adj': {}, 'others': {}}

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
						# uncovered_dict
						trigger = trigger.lower()
						if trigger not in uncovered_dict['verb']:
							uncovered_dict['verb'][trigger] = []
						uncovered_dict['verb'][trigger].append(sentence)
						# print(f"Verb not covered: {event['trigger']}\n{sentence}\n")
				elif all([pos[:2] == 'NN' for pos in poss[start:end]]):
					ace_trg_count['nom'] += 1
					if trigger in total_srl_preds:
						srl_trg_count['nom'] += 1
					else:
						# print(f"Nominal not covered: {event['trigger']}\n{sentence}\n")
						trigger = trigger.lower()
						if trigger not in uncovered_dict['nom']:
							uncovered_dict['nom'][trigger] = []
						uncovered_dict['nom'][trigger].append(sentence)
				elif all([pos[:2] == 'JJ' for pos in poss[start:end]]):
					ace_trg_count['adj'] += 1
					if trigger in noms or trigger in verbs:
						srl_trg_count['adj'] += 1
					else:
						# print(f"Adj not covered: {event['trigger']}\n{sentence}\n")
						# pass
						trigger = trigger.lower()
						if trigger not in uncovered_dict['adj']:
							uncovered_dict['adj'][trigger] = []
						uncovered_dict['adj'][trigger].append(sentence)
				else:
					ace_trg_count['others'] += 1
					if trigger in noms or trigger in verbs:
						srl_trg_count['adj'] += 1
					else:
						# print(f"Others not covered: {event['trigger']}\n{sentence}\n")
						# pass
						trigger = trigger.lower()
						if trigger not in uncovered_dict['others']:
							uncovered_dict['others'][trigger] = []
						uncovered_dict['others'][trigger].append(sentence)



	print('SRL trigger count:', srl_trg_count, '\nACE trigger count:', ace_trg_count)
	print("Number of ACE triggers covered:", sum([value for key,value in srl_trg_count.items()]))
	print("Number of ACE single-word triggers:",single_word_trg_count)
	print("Number of ACE triggers:", total_trg_count)


	fwn = f'data/srl_output/{srl_model_name}/coverage_info_{srl_model_name}.txt'

	sorted_word_dicts = []
	with open(fwn, 'w') as fw:
		for pos, word_dict in uncovered_dict.items():
			fw.write(f"POS: {pos}\n")
			sorted_word_dict = sorted(word_dict.items(), key=lambda x:len(x[1]), reverse=True)
			sorted_word_dicts.append(sorted_word_dict)
			word_count_list = [(word, len(sents)) for word, sents in sorted_word_dict]
			for i in range(len(word_count_list)):
				fw.write(f"{word_count_list[i][0]}: {word_count_list[i][1]}\n")
			fw.write("\n")

		for sorted_word_dict in sorted_word_dicts:
			for word, sents in sorted_word_dict:
				fw.write(word)
				fw.write("\n")
				for sent in sents:
					fw.write(sent)
					fw.write("\n")
				fw.write("\n")




