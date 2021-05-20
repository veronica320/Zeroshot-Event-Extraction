import requests
from pprint import pprint
import json
import os

def request_srl(sentence, srl_model_name):
	if srl_model_name == "illinois":
		data = {
			'text': sentence,
			'views': 'SRL_VERB,SRL_NOM'
		}
		response = requests.post('http://macniece.seas.upenn.edu:4001/annotate', data=data)

	elif srl_model_name in ["celine_new", "celine_new_all"]:
		input = '{"sentence": "' + sentence + '"}'

		headers = {'Content-Type': 'application/json'}
		port = ["http://leguin.seas.upenn.edu:4039/annotate",  # old port
		        "https://cogcomp.seas.upenn.edu/dc4039/annotate",
		        ][1]
		try:
			response = requests.post(port, headers=headers, data=input)
		except UnicodeEncodeError:
			return None, None, None


	text = response.text
	try:
		json_text = json.loads(text)
	except json.decoder.JSONDecodeError:
		print(f"SRL output is None: {sentence}")
		return None, None, None

	verb_srl_result, nom_srl_result = None, None

	verb_srl_view_name = {"illinois":"SRL_VERB",
	                      "celine_new":"SRL_ONTONOTES",
	                      "celine_new_all": "SRL_ONTONOTES",
	                      }

	nom_srl_view_name = {"illinois":"SRL_NOM",
	                     "celine_new":"SRL_NOM",
	                     "celine_new_all": "SRL_NOM_ALL"
	                     }

	words = json_text["tokens"]
	views = json_text["views"]

	for view in views:
		if view["viewName"] == verb_srl_view_name[srl_model_name]:
			verb_srl_result = view
		elif view["viewName"] == nom_srl_view_name[srl_model_name]:
			nom_srl_result = view

	return words, verb_srl_result, nom_srl_result


def extract_srl_contents(words, srl_result, predicate_type):
	if not srl_result:
		return []

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

def write_to_file(srl_output_list, fwn, mode):
	with open(fwn, mode) as fw:
		for item in srl_output_list:
			item_str = json.dumps(item)
			fw.write(item_str)
			fw.write("\n")


if __name__ == "__main__":

	root_dir = "/shared/lyuqing/probing_for_event/"
	os.chdir(root_dir)

	# custom configs
	input_dir = ["data/ACE_oneie/en/event_only", "data/ERE/ERE_oneIE/LDC2015E29"][1]
	input_split = ["dev", "E29"][1]
	srl_model_name = ["celine_old", "celine_new", "celine_new_all", "illinois"][2]

	input_fn = f"{input_dir}/{input_split}.event.json"

	verb_output = []
	nom_output = []

	output_dir = f"data/srl_output/{srl_model_name}"
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)

	verb_fwn = f"{output_dir}/verbSRL_{srl_model_name}_{input_split}.json"
	nom_fwn = f"{output_dir}/nomSRL_{srl_model_name}_{input_split}.json"

	with open(input_fn, 'r') as input_f:
		for i,line in enumerate(input_f):
			if i < 2060:
				continue

			if i % 50 == 0:
				print(f"{i} sentences finished.")

			# if i % 10 == 0:
			# 	write_to_file(verb_output, verb_fwn, mode='a')
			# 	write_to_file(nom_output, nom_fwn, mode='a')
			# 	verb_output = []
			# 	nom_output = []

			line = json.loads(line)
			sentence = line["sentence"]

			if "celine" in srl_model_name:
				sentence = sentence.replace('"', '\\"')

			srl_words, verb_srl_result, nom_srl_result = request_srl(sentence, srl_model_name)

			if not srl_words:
				srl_words = line["tokens"]

			verb_pred_arg_list = extract_srl_contents(srl_words, verb_srl_result, "verb")
			nom_pred_arg_list = extract_srl_contents(srl_words, nom_srl_result, "nominal")


			verb_output.append({"sent_id": line["sent_id"],
			                    "doc_id": line["doc_id"],
			                    "sentence": sentence,
			                    "verbs": verb_pred_arg_list,
			                    "words": srl_words
			                    })
			nom_output.append({"sent_id": line["sent_id"],
			                    "doc_id": line["doc_id"],
			                    "sentence": sentence,
			                    "nominals": nom_pred_arg_list,
			                    "words": srl_words
			                   })

		write_to_file(verb_output, verb_fwn, mode='a')
		write_to_file(nom_output, nom_fwn, mode='a')