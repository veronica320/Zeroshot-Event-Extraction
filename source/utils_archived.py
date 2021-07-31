import os
import json
import glob
import lxml.etree as et
from nltk import word_tokenize, sent_tokenize
from copy import deepcopy
from allennlp.predictors.predictor import Predictor
from pprint import pprint
import torch
import collections
import numpy as np
from itertools import product

bert_type_models = ['bert',
                    'bertl',
					'mbert',
                    'qamr-squad2_bert-l',
                    'qamr_mbert',
                    'qamr_mbert-cased',
                    "squad2_mbert",
					"MLQA_squad2_mbert",
					'elior_bert-lc_mnli',
					'elior_bert_squad2',
                    "elior_bert-lc_mnli_squad2",
					'squad2_elior_bert-lc_mnli',
					'qamr_elior_bert-lc_mnli',
					'qamr-squad2_elior_bert-lc_mnli',
                    'squad2_1sent_bert',
                    'squad2_1sent_roberta',
                    'squad2-s_na+ha_bert',
                    'squad2_ha_bert',
                    ]


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
	# print("\n")
	# print(pred)
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
	# print(bert_span)
	# print(gold_span)
	# print(answer_tokens)

	return {'span': gold_span,
	        'answer': answer,
	        'answer_tokens': answer_tokens,
	        'confidence': pred["confidence"],
	        "start_logit": pred["start_logit"],
	        "end_logit": pred["end_logit"],
	        }

def postprocess_qa_predictions(input_ids,
                               predictions, # Tuple[np.ndarray, np.ndarray],
							   question_len,
                               version_2_with_negative: bool = False,
                               n_best_size: int = 10,
                               max_answer_length: int = 30,
                               null_score_diff_threshold: float = 0.0,
                               ):
	"""
	Adapted from huggingface utils_qa.py.
	Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
	original contexts. This is the base postprocessing functions for models that only return start and end logits.
	Args:
		predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
			The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
			first dimension must match the number of elements of :obj:`features`.
		version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
			Whether or not the underlying dataset contains examples with no answers.
		n_best_size (:obj:`int`, `optional`, defaults to 20):
			The total number of n-best predictions to generate when looking for an answer.
		max_answer_length (:obj:`int`, `optional`, defaults to 30):
			The maximum length of an answer that can be generated. This is needed because the start and end predictions
			are not conditioned on one another.
		null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
			The threshold used to select the null answer: if the best answer has a score that is less than the score of
			the null answer minus this threshold, the null answer is selected for this example (note that the score of
			the null answer for an example giving several features is the minimum of the scores for the null answer on
			each feature: all features must be aligned on the fact they `want` to predict a null answer).
			Only useful when :obj:`version_2_with_negative` is :obj:`True`.
		output_dir (:obj:`str`, `optional`):
			If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
			:obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
			answers, are saved in `output_dir`.
		prefix (:obj:`str`, `optional`):
			If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
		is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
			Whether this process is the main process or not (used to determine if logging/saves should be done).
	"""

	prelim_predictions = []

	assert len(predictions) == 2, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
	start_logits, end_logits = predictions
	start_logits = [float(_) for _ in start_logits]
	end_logits = [float(_) for _ in end_logits]

	# Update minimum null prediction.
	null_score = start_logits[0] + end_logits[0]
	null_prediction = {
		"span": (0, 0),
		"answer": "",
		"confidence": null_score,
		"start_logit": start_logits[0],
		"end_logit": end_logits[0],
	}

	# Go through all possibilities for the `n_best_size` greater start and end logits.
	start_indices = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
	end_indices = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
	for start_index, end_index in product(start_indices, end_indices):
		# Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
		# to part of the input_ids that are not in the context.

		if start_index >= len(input_ids)-1 or end_index >= len(input_ids)-1 :
			continue

		# Don't add null prediction here.
		if start_index == 0 and end_index == 0:
			continue

		# Don't consider answers with a length that is either < 0 or > max_answer_length.
		if end_index < start_index or end_index - start_index + 1 > max_answer_length:
			continue

		# Answer includes tokens before the context
		if end_index <= question_len or start_index <= question_len:
			continue

		# Answer includes the last special token
		if start_index == len(input_ids) or end_index == len(input_ids):
			continue

		prelim_predictions.append(
			{
				"answer": "non-empty",
				"span": (start_index, end_index),
				"confidence": start_logits[start_index] + end_logits[end_index],
				"start_logit": start_logits[start_index],
				"end_logit": end_logits[end_index],
			}
		)

	if version_2_with_negative:
		# Add the minimum null prediction
		prelim_predictions.append(null_prediction)

	# Only keep the best `n_best_size` predictions.
	all_predictions = sorted(prelim_predictions, key=lambda x: x["confidence"], reverse=True)[:n_best_size]

	# Add back the minimum null prediction if it was removed because of its low score.
	if version_2_with_negative and not any(p["span"] == (0, 0) for p in all_predictions):
		all_predictions.append(null_prediction)

	# Use the offsets to gather the answer text in the original context.
	# for pred in all_predictions:
	# 	span = pred["span"]
	# 	start = span[0]
	# 	end = span[1] + 1
	# 	pred["answer"] = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[start:end]))

	# In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
	# failure.
	if len(all_predictions) == 0 or (len(all_predictions) == 1 and all_predictions[0]["answer"] == ""):
		all_predictions.insert(0, {"answer": "empty", "span": (0, 0), "start_logit": 0.0, "end_logit": 0.0, "confidence": 0.0})

	# Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
	# the LogSumExp trick).
	scores = np.array([pred.pop("confidence") for pred in all_predictions])
	exp_scores = np.exp(scores - np.max(scores))
	probs = exp_scores / exp_scores.sum()

	# Include the probabilities in our predictions.
	for prob, pred in zip(probs, all_predictions):
		pred["confidence"] = prob

	return all_predictions, null_prediction

def gold_to_bert_tokens(tokenizer, gold_tokens, EX_QA_model_type):
	"""Tokenize a piece of text using a Huggingface transformers tokenizer, and get a mapping between gold tokens and bert tokens. """
	goldid_2_bertid = []
	if EX_QA_model_type in bert_type_models:
		bert_tokens = []
		bertid_2_goldid = []
		grouped_inputs = []  # input ids to pass to QA model
	else:
		bert_tokens = ['<s>']
		bertid_2_goldid = [-1]
		grouped_inputs = [torch.LongTensor([tokenizer.bos_token_id])]  # input ids to pass to QA model

	for goldid, gold_token in enumerate(gold_tokens):
		goldid_2_bertid.append([])
		if EX_QA_model_type in bert_type_models:
			_tokens_encoded = tokenizer.encode(gold_token, return_tensors="pt", add_special_tokens=False).squeeze(axis=0)
		elif EX_QA_model_type == 'qamr_xlm-roberta':
			_tokens_encoded = tokenizer.encode(gold_token, return_tensors="pt", add_special_tokens=False).squeeze(axis=0)
		else:
			_tokens_encoded = tokenizer.encode(gold_token, add_prefix_space=True, return_tensors="pt", add_special_tokens=False).squeeze(axis=0)
		_tokens = tokenizer.convert_ids_to_tokens(_tokens_encoded.tolist())
		grouped_inputs.append(_tokens_encoded)
		for bert_token in _tokens:
			bert_tokens.append(bert_token)
			bertid_2_goldid.append(goldid)
			goldid_2_bertid[-1].append(len(bertid_2_goldid) - 1)
	if EX_QA_model_type in bert_type_models:
		grouped_inputs.append(torch.LongTensor([tokenizer.sep_token_id]))  # input ids to pass to QA model
		bert_tokens.append('[SEP]')
	else:
		grouped_inputs.append(torch.LongTensor([tokenizer.eos_token_id]))
		bert_tokens.append('</s>')
	bertid_2_goldid.append(-1)
	flattened_inputs = torch.cat(grouped_inputs)
	flattened_inputs = torch.unsqueeze(flattened_inputs, 0)
	return flattened_inputs, bert_tokens, goldid_2_bertid, bertid_2_goldid


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

def find_lowest_constituent(predictor, trigger_text, sent):
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
							node['type'] in ['NP','PP', 'S']:
				return node['word']


