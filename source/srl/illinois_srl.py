import requests
from pprint import pprint
import json
import os

def request_srl(sentence):
	input = '{"sentence": "' + sentence + '"}'

	headers = {'Content-Type': 'application/json'}
	response = requests.post('http://leguin.seas.upenn.edu:4039/annotate', headers=headers, data=input)
	text = response.text
	try:
		json_text = json.loads(text)
	except json.decoder.JSONDecodeError:
		print(f"SRL output is None: {sentence}")
		return None, None, None

	words = json_text["tokens"]

	views = json_text["views"]
	verb_srl_result = views[2]
	assert(verb_srl_result["viewName"] == "SRL_ONTONOTES")
	nom_srl_result = views[3]
	assert(nom_srl_result["viewName"] == "SRL_NOM")

	return words, verb_srl_result, nom_srl_result


def extract_srl_contents(words, srl_result, predicate_type):
	view_data = srl_result["viewData"][0]
	constituents = view_data["constituents"]

	pred_arg_list = []

	for const in constituents:
		start = const["start"]
		end = const["end"]
		label = const["label"]

		# a new predicate-argument group
		if label == "Predicate":
			predicate = ' '.join(words[start:end])
			pred_arg_list.append({predicate_type: predicate,
			                      "tags": ['O'] * len(words)
			                      })
			pred_arg_list[-1]["tags"][start] = f"B-V"
			for i in range(start + 1, end):
				pred_arg_list[-1]["tags"][i] = f"I-V"

		# an argument of the current predicate
		else:
			pred_arg_list[-1]["tags"][start] = f"B-{label}"
			for i in range(start+1, end):
				pred_arg_list[-1]["tags"][i] = f"I-{label}"

	return pred_arg_list

def write_to_file(srl_output_list, fwn):
	with open(fwn, 'w') as fw:
		for item in srl_output_list:
			item_str = json.dumps(item)
			fw.write(item_str)
			fw.write("\n")


if __name__ == "__main__":

	root_dir = "/shared/lyuqing/probing_for_event/"
	os.chdir(root_dir)

	input_dir = "data/ACE_oneie/en/event_only"
	input_split = "dev"
	input_fn = f"{input_dir}/{input_split}.event.json"
	verb_output = []
	nom_output = []

	with open(input_fn, 'r') as input_f:
		i = 0
		for line in input_f:

			line = json.loads(line)

			sentence = line["sentence"]
			sentence_w_slash = sentence.replace('"', '\\"')

			words, verb_srl_result, nom_srl_result = request_srl(sentence_w_slash)
			if words == None:
				continue

			verb_pred_arg_list = extract_srl_contents(words, verb_srl_result, "verb")
			nom_pred_arg_list = extract_srl_contents(words, nom_srl_result, "nominal")

			verb_output.append({"sent_id": line["sent_id"],
			                    "doc_id": line["doc_id"],
			                    "sentence": sentence,
			                    "verbs": verb_pred_arg_list,
			                    "words": words
			                    })
			nom_output.append({"sent_id": line["sent_id"],
			                    "doc_id": line["doc_id"],
			                    "sentence": sentence,
			                    "nominals": nom_pred_arg_list,
			                    "words": words
			                   })
			if i % 50 == 0:
				print(f"{i} sentences finished.")

			i += 1


	output_dir = "data/srl_output/illinois_srl"
	verb_fwn = f"{output_dir}/verbSRL_illinois_{input_split}.json"
	nom_fwn = f"{output_dir}/nomSRL_illinois_{input_split}.json"

	write_to_file(verb_output, verb_fwn)
	write_to_file(nom_output, nom_fwn)




