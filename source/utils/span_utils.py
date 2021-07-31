## Utils to fix the inconsistency between spans of different tokenization strategies. ##
import torch

bert_type_models = ["bert", "bertl"]


def find_lowest_constituent(predictor, trigger_text, sent):
	'''Find the lowest constituent above the current trigger.'''

	pred = predictor.predict(sentence=sent)
	root = pred['hierplane_tree']['root']
	cur_node = root

	parent_level = 0
	level_stack = [[root]]
	if_still_child = True

	while if_still_child:
		if_still_child = False
		for node in level_stack[-1]:
			if 'children' in node:
				if_still_child = True
				if len(level_stack) == parent_level + 1:
					level_stack.append([])
				for child in node['children']:
					level_stack[-1].append(child)
		parent_level += 1
	slim_level_stack = []
	for level in level_stack:
		slim_level_stack.append([])
		for node in level:
			slim_level_stack[-1].append({'word': node['word'],
			                             'type': node['nodeType']})

	for level_id in range(len(slim_level_stack) - 1, -1, -1):
		level = slim_level_stack[level_id]
		for node in level:
			if (trigger_text in node['word'] or node['word'] in trigger_text) and \
							' ' in node['word'] and \
							node['type'] in ['NP', 'PP', 'S']:
				return node['word']


def get_head(dependency_parser, span, tokens, pos_tags):
	""" A coarse-grained head identifier. """

	instance = dependency_parser._dataset_reader.text_to_instance(tokens, pos_tags)
	output = dependency_parser.predict_instance(instance)

	start_ix = span[0]

	root_idx = output['predicted_heads'].index(0)
	pos_list = output['pos']
	words_list = output['words']
	parent_idx = 0
	current_idx = root_idx
	siblings = [current_idx]
	pos_in_siblings = 0
	while True:
		if pos_list[current_idx].startswith(('NN', 'PRP', 'CD')):
			word = words_list[current_idx]
			global_idx = start_ix + current_idx
			return global_idx, word
		pos_in_siblings = pos_in_siblings - 1
		if pos_in_siblings >= 0:  # check if there are siblings on the left of the current node
			current_idx = siblings[pos_in_siblings]  # if yes, move to the rightmost sibling
		else:  # if no, move to the rightmost child of the current node
			parent_idx = current_idx + 1
			siblings = [i for i, x in enumerate(output['predicted_heads']) if x == parent_idx]
			if siblings:
				current_idx = siblings[-1]
				pos_in_siblings = len(siblings) - 1
			else:
				return None, None

def gold_to_bert_tokens(tokenizer, gold_tokens, QA_model_type):
	"""Tokenize a piece of text using a Huggingface transformers tokenizer, and get a mapping between gold tokens and bert tokens. """

	goldid_2_bertid = []

	## bert type models
	if QA_model_type in bert_type_models:
		bert_tokens = []
		bertid_2_goldid = []
		grouped_inputs = []  # input ids to pass to QA model
	else:
		bert_tokens = ['<s>']
		bertid_2_goldid = [-1]
		grouped_inputs = [torch.LongTensor([tokenizer.bos_token_id])]  # input ids to pass to QA model

	for goldid, gold_token in enumerate(gold_tokens):
		goldid_2_bertid.append([])
		if QA_model_type in bert_type_models:
			_tokens_encoded = tokenizer.encode(gold_token, return_tensors="pt", add_special_tokens=False).squeeze(axis=0)
		else:
			_tokens_encoded = tokenizer.encode(gold_token, add_prefix_space=True, return_tensors="pt", add_special_tokens=False).squeeze(axis=0)
		_tokens = tokenizer.convert_ids_to_tokens(_tokens_encoded.tolist())
		grouped_inputs.append(_tokens_encoded)
		for bert_token in _tokens:
			bert_tokens.append(bert_token)
			bertid_2_goldid.append(goldid)
			goldid_2_bertid[-1].append(len(bertid_2_goldid) - 1)
	if QA_model_type in bert_type_models:
		grouped_inputs.append(torch.LongTensor([tokenizer.sep_token_id]))  # input ids to pass to QA model
		bert_tokens.append('[SEP]')
	else:
		grouped_inputs.append(torch.LongTensor([tokenizer.eos_token_id]))
		bert_tokens.append('</s>')
	bertid_2_goldid.append(-1)
	flattened_inputs = torch.cat(grouped_inputs)
	flattened_inputs = torch.unsqueeze(flattened_inputs, 0)

	return flattened_inputs, bert_tokens, goldid_2_bertid, bertid_2_goldid


def match_bert_span_to_text(pred,
			                bertid_2_goldid,
			                question_len,
			                context_tokens
                            ):
	"""
		Match the predicted bert span to the text in gold context.
		Args:
			pred (:obj:`dict`):
				The prediction dictionary.
			bertid_2_goldid (:obj:`list`):
				The list mapping bert token ids to gold token ids.
			question_len (:obj:`int`):
				The length of the question in bert tokens.
			context_tokens (:obj:`list`):
				The list of gold context tokens.
		"""

	answer_start, answer_end = pred["span"]

	# null prediction
	if (answer_start, answer_end) == (0, 0):
		return {'span': None,
		        'answer': None,
		        'answer_tokens': None,
		        'confidence': pred["confidence"],
		        "start_logit": pred["start_logit"],
		        "end_logit": pred["end_logit"],
		        }

	# prediction is not in context
	if (answer_start < question_len or answer_end < question_len):
		return None

	bert_span = (answer_start - question_len, answer_end - question_len)  # span in bert tokens

	gold_span = (bertid_2_goldid[bert_span[0]], bertid_2_goldid[bert_span[1]] + 1)  # span in gold tokens

	# span contains invalid tokens
	if (gold_span[0] < 0 or gold_span[1] < 0):
		return None

	answer_tokens = context_tokens[gold_span[0]:gold_span[1]]

	answer = ' '.join(answer_tokens)

	return {'span': gold_span,
	        'answer': answer,
	        'answer_tokens': answer_tokens,
	        'confidence': pred["confidence"],
	        "start_logit": pred["start_logit"],
	        "end_logit": pred["end_logit"],
	        }

