## Utils for loading the srl results. ##

from nltk.corpus import stopwords
import json

def load_srl(input_file):
	"""Loads cached SRL predictions for an input file."""

	verb_srl_dict, nom_srl_dict = {}, {}

	if "ACE" in input_file:
		dataset = "ACE"
	elif "ERE" in input_file:
		dataset = "ERE"
	else:
		raise ValueError("Unknown dataset")

	split = input_file.split('/')[-1].split('.')[0]

	for type in ['verb', 'nom']:
		path = f"data/SRL_output/{dataset}/{type}SRL_{split}.jsonl"
		with open(path, 'r') as fr:
			for line in fr:
				srl_res = json.loads(line)
				sent_id = srl_res["sent_id"]
				if type == 'nom':
					nom_srl_dict[sent_id] = {"nominals": srl_res["nominals"],
					                         "words": srl_res["words"]
					                         }
				if type == 'verb':
					verb_srl_dict[sent_id] = {"verbs": srl_res["verbs"],
					                          "words": srl_res["words"]
					                          }
	return verb_srl_dict, nom_srl_dict

def load_stopwords():
	"""Load NLTK stopwords. """

	sw = stopwords.words('english')
	sw += ['said', 'say', 'says', 'saying', 'want', 'wants', 'wanted']
	return sw

def overlap(tag, srl_args):
	"""Checks if a tag is in the set of SRL tags to include in the textpiece.

	:param tag (str): a pos tag from SRL output, e.g. 'B-V'.
	:param srl_args (list): a list of SRL tags to include in the textpiece set in the config, e.g. ['V', 'A1'].
	:return (bool): a boolean indicating if tag is in srl_args.
	"""
	flag = False
	if srl_args == 'all':
		if tag != 'O':
			flag = True
	else:
		tag = tag.split('-')
		for srl_arg in srl_args:
			if srl_arg in tag:
				flag = True
				break
	return flag

def get_gold_map(tokens, gold_tokens):
	"""There is often an inconsistency between arbitrary token ids (e.g. the SRL token ids) and the gold ACE token ids. This method maps arbitrary ids to gold ids.

	:param tokens (list): a list of arbitrary tokens.
	:param gold_tokens (list): a list of gold tokens.
	:return (list): a list mapping arbitrary token ids to gold token ids, i.e. tokenid_2_goldid[an_arbitrary_id] would give the corresponding gold id.
	"""
	tokenid_2_goldid = {}
	i, j = -1, -1  # token pointer, gold token pointer
	prefix_i = prefix_j = ''
	len_prefix_i = len_prefix_j = 0
	loop_count = 0
	while i < len(tokens) and j < len(gold_tokens):
		loop_count += 1
		if loop_count >= 1000:
			print(f'Infinite loop in finding gold map:{loop_count}\n{tokens}\n{gold_tokens}\n{prefix_i}\n{prefix_j}')
			break
		# return None
		if prefix_i == '':
			i += 1
			prefix_i += tokens[i]
		if prefix_j == '':
			j += 1
			prefix_j += gold_tokens[j]
		if prefix_i == prefix_j:  # matched
			for idx in range(i - len_prefix_i, i + 1):
				tokenid_2_goldid[idx] = j
			prefix_i = prefix_j = ''
			len_prefix_i = len_prefix_j = 0
			if i == len(tokens) - 1 and j == len(gold_tokens) - 1:
				break
		elif prefix_i in prefix_j:
			i += 1
			prefix_i += tokens[i]
			len_prefix_i += 1
		elif prefix_j in prefix_i:
			j += 1
			prefix_j += gold_tokens[j]
			len_prefix_j += 1
	assert [i in tokenid_2_goldid for i in range(len(tokens))]
	return tokenid_2_goldid

def get_srl_result_for_instance(srl_dict, instance):
	"""Get SRL output for an instance."""

	sent_id = instance.sent_id
	tokens_gold = instance.tokens
	srl_output = srl_dict[sent_id]
	srl_output["words"] = [word for word in srl_output["words"] if word != "\\"]
	tokens_srl = srl_output['words']
	if tokens_srl != tokens_gold:
		srl2gold_id_map = get_gold_map(tokens_srl, tokens_gold)
	else:
		srl2gold_id_map = {i: i for i in range(len(tokens_srl))}
	return srl_output, srl2gold_id_map

def get_srl_results(instance,
                    srl_dicts,
                    stopwords,
                    srl_consts,
                    ):
	"""Get the SRL result, text pieces, token maps for one instance. """

	srl_id_results = {}  # verb+nom srl results. Each item stores the result of a predicate. Format: {srl_id: srl_result, ....}
	text_pieces = {}  # pieces of text from the input sentence (e.g. concatenation of V, A0, A1) as the premise. Format: {srl_id: text_piece, ....}
	trg_cands = {}  # trigger candidates. Format: {srl_id: ((span_start, span_end), trigger_text), ....}
	srl2gold_maps = [] # mapping from srl tokens to gold tokens
	srl_id = 0  # a common key used across srl_id_results, text_pieces, verbs

	verb_srl_dict, nom_srl_dict = srl_dicts


	## Load verbSRL results

	# get mapping from srl tokens to gold tokens
	verb_srl_output, verb_srl2gold = get_srl_result_for_instance(verb_srl_dict, instance)  # the entire srl output; mapping from SRL token ids to gold token ids
	verb_srl_tokens, verb_srl_results = verb_srl_output['words'], verb_srl_output['verbs']  # tokens according to SRL tokenization; srl results
	srl2gold_maps.append(verb_srl2gold)

	# get non-stopword verbs
	if not verb_srl_results:
		verb_srl_results = []
	for res in verb_srl_results:
		if set(res['tags']) != {'B-V', 'O'} and res['verb'] not in stopwords:
			span = [i for i, tag in enumerate(res['tags']) if tag in ['B-V', 'I-V']]
			if span:
				span = (verb_srl2gold[span[0]], verb_srl2gold[span[-1]] + 1)  # map srl ids to gold ids
				text_piece = ' '.join([verb_srl_tokens[i] for i, tag in enumerate(res['tags']) if
				                       overlap(tag, srl_consts)])  # get the text piece as the concatenation of the SRL predicate and certain arguments
				text_pieces[srl_id] = text_piece
				trg_cands[srl_id] = (span, res['verb'])
				srl_id_results[srl_id] = res
				srl_id_results[srl_id]['predicate_type'] = 'verb'
				srl_id_results[srl_id]['words'] = verb_srl_output['words']
				srl_id += 1

	## Load nomSRL results

	# get mapping from srl tokens to gold tokens
	nom_srl_output, nom_srl2gold = get_srl_result_for_instance(nom_srl_dict, instance)  # the entire srl output; mapping from SRL token ids to gold token ids
	nom_srl_tokens, nom_srl_results = nom_srl_output['words'], nom_srl_output['nominals']  # tokens according to SRL tokenization; srl results
	srl2gold_maps.append(nom_srl2gold)

	# get non-stopword nominals
	if not nom_srl_results:
		nom_srl_results = []
	for res in nom_srl_results:
		if set(res['tags']) != {'O'} and res['nominal'] not in stopwords:
			span = [i for i, tag in enumerate(res['tags']) if tag in ['B-V', 'I-V']]
			if span:
				span = (nom_srl2gold[span[0]], nom_srl2gold[span[-1]] + 1)  # map srl ids to gold ids
				text_piece = ' '.join([nom_srl_tokens[i] for i, tag in enumerate(res['tags']) if
				                       overlap(tag, srl_consts)])  # get the text piece as the concatenation of the SRL predicate and certain arguments
				text_pieces[srl_id] = text_piece
				trg_cands[srl_id] = (span, res['nominal'])
				srl_id_results[srl_id] = res
				srl_id_results[srl_id]['predicate_type'] = 'nom'
				srl_id_results[srl_id]['words'] = nom_srl_output['words']
				srl_id += 1

	return srl_id_results, text_pieces, trg_cands, srl2gold_maps
