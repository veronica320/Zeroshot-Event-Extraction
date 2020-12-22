# QA pipeline

import numpy as np
import pickle
import re
import os
import allennlp
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize, pos_tag
from allennlp.predictors.predictor import Predictor
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
import torch
from collections import OrderedDict
import json
from pprint import pprint
import ipdb
import sys
from utils import *

os.chdir('/shared/lyuqing/probing_for_event/')

model_abbr_dict = {"bert":"bert",
                   "bertl":"bert-l",
                   "roberta":"roberta",
                   "robertal":"roberta-l",
                   "xlmr":"xlm-roberta",
                   "xlmrl": "xlm-roberta-l",
                   "mbert":"mbert",
                   "mbert-c":"mbert-cased"
                   }


class EventDetectorQA():
	def __init__(self,
	             config,
	             ):
		self.cache_dir = config.bert_cache_dir

		self.YN_QA_model_type = config.YN_QA_model_type # Yes/No QA model
		self.YN_idk = config.YN_idk # IDK class in Yes/No QA model

		self.EX_QA_model_type = config.YN_QA_model_type # Extractive QA model
		self.EX_idk = config.EX_idk # IDK class in Extractive QA model

		if config.use_gpu and config.gpu_devices != -1:
			self.gpu_devices = [int(_) for _ in config.gpu_devices.split(",")]
		else:
			self.gpu_devices = None

		self.classification_only = config.classification_only
		self.gold_trigger = config.gold_trigger

		self.srl_args = eval(config.srl_args)
		self.trg_thresh = eval(config.trg_thresh)
		self.arg_thresh = eval(config.arg_thresh)
		self.predicate_type = eval(config.predicate_type)
		self.add_neutral = config.add_neutral
		self.identify_head = config.identify_head
		self.tune_on_gdl = eval(config.tune_on_gdl)
		self.const_premise = eval(config.const_premise)
		self.pair_premise_strategy = eval(config.pair_premise_strategy)

		self.arg_probe_type = eval(config.arg_probe_type)

		# Load trigger probes
		probe_dir = 'source/lexicon/probes/'
		trg_probes_frn = f'{probe_dir}trg_qa_probes_fine.txt' # TODO: other types of probes
		with open(trg_probes_frn, 'r') as fr:
			self.trg_probe_lexicon = load_trg_probe_lexicon(fr)

		# Load argument probes and the SRL-to-ACE argument map
		if self.arg_probe_type == 'bool':
			arg_probes_frn = f'{probe_dir}arg_qa_probes_bool.txt'
		elif self.arg_probe_type == 'ex':
			arg_probes_frn = f'{probe_dir}arg_qa_probes_ex.txt'
		elif self.arg_probe_type == 'ex_wtrg': # instantiated trigger in the argument questions
			arg_probes_frn = f'{probe_dir}arg_qa_probes_ex_wtrg.txt'
		with open(arg_probes_frn, 'r') as fr:
			self.arg_probe_lexicon = load_arg_probe_lexicon(fr, self.arg_probe_type)
		with open('source/lexicon/arg_srl2ace.txt') as fr:
			self.arg_map = load_arg_map(fr)

		# Event types
		self.trg_subtypes = self.trg_probe_lexicon.keys()

		# Load stopwords that will be exluded from SRL predicates as potential triggers
		self.sw = load_stopwords()

		# Load cached SRL output
		self.verb_srl_dict, self.nom_srl_dict = load_srl()

	def load_models(self):
		print('Loading constituency and dependency parser...')
		self.dependency_parser = Predictor.from_path(
			"https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")
		self.constituency_parser = Predictor.from_path(
			"https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")

		model_dir = "output_model_dir"
		YN_model_name = model_abbr_dict[self.YN_QA_model_type]
		EX_model_name = model_abbr_dict[self.EX_QA_model_type]
		yn_qa_model_path = f"{model_dir}/boolq{'_idk' if self.YN_idk else ''}_{YN_model_name}" # Yes/No QA model
		ex_qa_model_path = f"{model_dir}/qamr{'_idk' if self.EX_idk else ''}_{EX_model_name}" # Extractive QA model

		print(f'Loading QA models...')
		if self.gpu_devices:
			self.yn_qa_model = AutoModelForSequenceClassification.from_pretrained(yn_qa_model_path
			                                                                      ).to('cuda:' + str(self.gpu_devices[0]))
			self.ex_qa_model = AutoModelForQuestionAnswering.from_pretrained(ex_qa_model_path
			                                                                 ).to('cuda:' + str(self.gpu_devices[0]))
		else:
			self.yn_qa_model = AutoModelForSequenceClassification.from_pretrained(yn_qa_model_path)
			self.ex_qa_model = AutoModelForQuestionAnswering.from_pretrained(ex_qa_model_path)

		self.yn_tokenizer = AutoTokenizer.from_pretrained(yn_qa_model_path)
		self.ex_tokenizer = AutoTokenizer.from_pretrained(ex_qa_model_path)

	def predict(self, instance):
		"""Predict on a single instance.
		:param instance (Instance): a sentence instance
		"""

		srl_id_results, text_pieces, trg_cands, srl2gold_maps = get_srl_results(instance,
		                                                                        self.predicate_type,
		                                                                        (self.verb_srl_dict, self.nom_srl_dict),
		                                                                        self.sw,
		                                                                        self.srl_args
		                                                                        )  # get SRL results for the current instance
		pred_events = []  # a list of predicted events

		# predict triggers
		pred_events = self.extract_triggers(instance, pred_events, srl_id_results, text_pieces, trg_cands)

		# # predict arguments
		pred_events = self.extract_arguments(instance, pred_events, srl_id_results, text_pieces, trg_cands, srl2gold_maps)

		return pred_events

	def extract_triggers(self, instance, pred_events, srl_id_results, text_pieces, trg_cands):
		"""Extract triggers."""

		sent = instance.sentence
		tokens_gold = instance.tokens  # ACE tokens

		if self.gold_trigger: # directly return gold trigger identification + classification results
			for event in instance.events:
				gold_trg_res = {"event_type": event['event_type'],
				                "trigger": event["trigger"].copy(),
				                "arguments": []}
				pred_events.append(gold_trg_res)

			for event_id, event in enumerate(pred_events):  # Get the context from SRL (for argument extraction)
				trigger_text = event["trigger"]["text"]

				# Get the context
				srl_id, text_piece = None, None
				for id, cand in trg_cands.items():
					if trigger_text in cand[1] or cand[1] in trigger_text:  # if SRL predicate overlaps with the gold trigger
						text_piece = text_pieces[id]  # Use the srl text piece as the premise
						srl_id = id
				if text_piece == None:  # if the gold trigger isn't in SRL prediates
					if self.const_premise == 'whenNone':  # use the lowest constituent as the premise
						text_piece = find_lowest_constituent(self.constituency_parser, trigger_text, sent)
				if self.const_premise == 'alwaystrg':  # regardless of whether the gold trigger is in SRL predicates, always use the lowest constituent as the premise
					text_piece = find_lowest_constituent(self.constituency_parser, trigger_text, sent)

				context = text_piece if text_piece else sent  # if text_piece is None, use the entire sentence as the premise

				pred_events[event_id]["text_piece"] = text_piece
				pred_events[event_id]['srl_id'] = srl_id

		elif self.classification_only:  # do trigger classification only
			# Get the gold identified triggers
			for event in instance.events:
				gold_trg_res = {"event_type": None,
				                "trigger": event["trigger"].copy(),
				                "arguments": []}
				pred_events.append(gold_trg_res)

			for event_id, event in enumerate(pred_events):  # Classify each gold trigger
				trigger_text = event["trigger"]["text"]

				# Get the premise
				srl_id, text_piece = None, None
				for id, cand in trg_cands.items():
					if trigger_text in cand[1] or cand[1] in trigger_text:  # if SRL predicate overlaps with the gold trigger
						text_piece = text_pieces[id]  # Use the srl text piece as the premise
						srl_id = id
				if text_piece == None:  # if the gold trigger isn't in SRL prediates
					if self.const_premise == 'whenNone':  # use the lowest constituent as the premise
						text_piece = find_lowest_constituent(self.constituency_parser, trigger_text, sent)
				if self.const_premise == 'alwaystrg':  # regardless of whether the gold trigger is in SRL predicates, always use the lowest constituent as the premise
					text_piece = find_lowest_constituent(self.constituency_parser, trigger_text, sent)

				context = text_piece if text_piece else sent  # if text_piece is None, use the entire sentence as the premise

				top_type, confidence = self.classify_a_trigger(context, trigger_text)

				pred_events[event_id]["event_type"] = top_type
				pred_events[event_id]["text_piece"] = text_piece
				pred_events[event_id]["trigger"]['confidence'] = confidence
				pred_events[event_id]['srl_id'] = srl_id

		else:  # do trigger identification + classification
			for srl_id, text_piece in text_pieces.items():
				trigger_text = trg_cands[srl_id][1]
				context = text_piece

				top_type, confidence = self.classify_a_trigger(context, trigger_text)

				if confidence > self.trg_thresh:
					event = {'event_type': top_type,
							 'text_piece': text_piece,
							 'trigger': {'text': trg_cands[srl_id][1],
										 'start': trg_cands[srl_id][0][0],
										 'end': trg_cands[srl_id][0][1],
										 'confidence': confidence,
										 },
							 'arguments': [],
							 'srl_id': srl_id,
							 }
					pred_events.append(event)

		return pred_events

	def extract_arguments(self, instance, pred_events, srl_id_results, text_pieces, trg_cands, srl2gold_maps):
		"""Extract arguments."""

		sent = instance.sentence
		tokens_gold = instance.tokens  # ACE tokens
		verb_srl2gold, nom_srl2gold = srl2gold_maps

		if self.classification_only:
			for gold_event, pred_event in zip(instance.events, pred_events):
				pred_event['arguments'] = [{"text": arg["text"],
				                            "role": None,
				                            "start": arg["start"],
				                            "end": arg["end"]}
				                           for arg in gold_event["arguments"]]

			for event_id, event in enumerate(pred_events):
				srl_id = event['srl_id']
				trigger_text = event['trigger']['text']
				event_type = event['event_type']

				# Get the context
				text_piece = None
				if srl_id: # if the gold trigger is in the SRL predicates
					srl_result = srl_id_results[srl_id]
					srl_tokens = srl_result['words']
					text_piece = ' '.join([srl_tokens[i] for i, tag in enumerate(srl_result['tags']) if tag != 'O']) # Concatenate all SRL arguments as the premise
				context = text_piece if text_piece else sent

				# Classify each argument
				cand_ace_args = self.arg_probe_lexicon[event_type]  # Take all ACE argument types of the current event type as candidates
				for arg_id, arg in enumerate(event["arguments"]):
					top_arg_name, top_arg_score = self.classify_an_argument(arg, event_type, context, cand_ace_args)
					event["arguments"][arg_id]['role'] = top_arg_name
					event["arguments"][arg_id]['confidence'] = top_arg_score

		else:
			for event_id, event in enumerate(pred_events):
				srl_id = event['srl_id']
				if self.gold_trigger and srl_id == None and self.arg_probe_type == 'bool': # the gold trigger isn't in SRL predicates. YN QA can't identify arguments.
					continue
				trigger_text = event['trigger']['text']
				event_type = event['event_type']

				## Get the context
				# if srl_id: # TODO: add back
				# 	srl_result = srl_id_results[srl_id]
				# 	srl_tokens = srl_result['words']
				# 	text_piece_tokens = [srl_tokens[i] for i, tag in enumerate(srl_result['tags']) if tag != 'O']
				# 	text_piece = ' '.join(text_piece_tokens)
				# 	srl2gold = nom_srl2gold if srl_result['predicate_type'] == 'nom' else verb_srl2gold
				# 	context_tokens = text_piece_tokens
				# else:
				context_tokens = tokens_gold

				if self.arg_probe_type == 'bool': # Y/N quesitons
					# Construct srl_arg_dict
					srl_arg_dict = {}  # The span and tokens of all SRL arguments. Format: {'ARG0': [(span, token),(span, token)], 'ARG1': [(span, token), ...], ...}
					tag_set = set([tag[2:] for tag in srl_result['tags'] if tag not in ['O', 'B-V', 'I-V']]) # SRL argument tags: ARG0, ARG1, ARGM-TMP...
					for target_tag in tag_set:
						span = [j for j, tag in enumerate(srl_result['tags']) if tag[2:] == target_tag]  # TODO: multiple args for the same arg type
						tokens = [word for i, word in enumerate(srl_tokens) if i in span]
						if self.identify_head: # only retain the head
							try:
								pos_tags = [tag for _, tag in pos_tag(tokens)]
							except IndexError: # event 774: IndexError: string index out of range
								span = [None]
								continue
							head_idx, token = get_head(self.dependency_parser, span, tokens, pos_tags)
							span = [head_idx]
						else: # retain the whole SRL argument
							token = ' '.join([word for i, word in enumerate(srl_tokens) if i in span])
						if None not in span:
							span = (srl2gold[span[0]], srl2gold[span[-1]] + 1) # map SRL ids to gold ids
							if target_tag not in srl_arg_dict:
								srl_arg_dict[target_tag] = []
							srl_arg_dict[target_tag].append((span, token))

					# Classify each SRL argument
					for srl_arg_type, srl_arg_ists in srl_arg_dict.items():
						if srl_arg_type not in self.arg_map[event_type]: # the SRL argument isn't a potential ACE argument
							continue
						cand_ace_args = self.arg_map[event_type][srl_arg_type] # Only take the ACE argument types in the SRL-to-ACE argument mapping as candidates
						for srl_arg_ist in srl_arg_ists:  # an instance of SRL argument
							top_arg_name, top_arg_score = self.classify_an_argument(srl_arg_ist, event_type, text_piece, cand_ace_args)
							if top_arg_score >= self.arg_thresh:
								event['arguments'].append({'text': srl_arg_ist[1],
														   'role': top_arg_name,
														   'start': srl_arg_ist[0][0],
														   'end': srl_arg_ist[0][1],
														   'confidence': top_arg_score,
														  })
				elif self.arg_probe_type.startswith('ex'): # Extractive questions
					for cand_ace_arg in self.arg_probe_lexicon[event_type]:
						question = self.arg_probe_lexicon[event_type][cand_ace_arg]
						if self.arg_probe_type == 'ex_wtrg':
							question = question.replace('{trigger}', trigger_text)
						output = self.answer_ex(question, context_tokens)
						span = output['span']
						confidence = output["confidence"]
						if span and confidence >= self.arg_thresh: # top answer is not None; confidence is high enough
							answer_tokens = output['answer_tokens']
							arg_text = ' '.join(answer_tokens)
							if self.identify_head: # get the head
								pos_tags = [tag for _, tag in pos_tag(answer_tokens)]
								head_idx, head_token = get_head(self.dependency_parser, span, answer_tokens, pos_tags)
								if head_idx: # TODO: check when head is None
									span = (head_idx, head_idx+1)
									arg_text = head_token
							event['arguments'].append({'text': arg_text,
							                           'role': cand_ace_arg[:-4],
							                           'start': span[0],
							                           'end': span[1],
							                           'confidence': confidence,
							                           })
						else: # no answer; or confidence isn't high enough
							continue

		return pred_events

	def classify_a_trigger(self, context, trigger_text):
		"""Classify a single trigger."""

		result_dict = {}  # the temporary result dict for all possible event types. Format: {event_type:confidence_score, ...}
		for event_type in self.trg_subtypes:
			question = self.trg_probe_lexicon[event_type]  # the probe from the lexicon

			orig_yes_prob = self.answer_yn(question,
			                               context)  # original probability of yes
			if self.pair_premise_strategy:  # use a minimal pair of premises
				sub_pattern = '\s?' + trigger_text + '\s?'
				truncated_context = re.sub(pattern=sub_pattern, string=context,
				                           repl=' ').strip()  # the truncated context is the original context - the trigger
				truncated_yes_prob = self.answer_yn(question, truncated_context)  # the probability of yes given the truncated context
				delta = orig_yes_prob - truncated_yes_prob  # the difference

			if self.pair_premise_strategy == 'max_delta':
				result_dict[event_type] = delta  # maximizing delta
			elif self.pair_premise_strategy == 'max_conf+delta':
				result_dict[event_type] = orig_yes_prob + delta  # maximizing the sum of the original entailment prob + delta
			elif self.pair_premise_strategy == None:
				result_dict[event_type] = orig_yes_prob  # maximizing the original entailment prob

		sorted_res = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
		top_type, confidence = sorted_res[0][0], sorted_res[0][1]  # Get the top event type and its confidence score

		return top_type, confidence

	def classify_an_argument(self, arg, event_type, context, cand_ace_args):
		"""Classify a single argument."""

		cand_scores = {cand: 0 for cand in cand_ace_args}
		arg_text = arg['text'] if 'text' in arg else arg[1]
		for cand_ace_arg in cand_ace_args:
			if cand_ace_arg not in self.arg_probe_lexicon[event_type]: # TODO: check why cand_ace_arg can be None
				continue
			question = self.arg_probe_lexicon[event_type][cand_ace_arg]
			question = question.replace('{}', arg_text)
			confidence = self.answer_yn(question, context)
			cand_scores[cand_ace_arg] = confidence

		sorted_cands = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
		top_cand = sorted_cands[0]
		top_arg_name, top_arg_score = top_cand[0][:-4], top_cand[1]

		return top_arg_name, top_arg_score

	def predict_batch(self, batch):
		# TODO
		"""Predict on a batch of instances.
		:param batch (list): a list of Instance objects
		"""
		return None

	def predict_dataset(self, dataset):
		"""Predict on a dataset.
		:param dataset (list): a list of Instance objects.
		"""
		outputs = []
		for instance in dataset:
			output = self.predict(instance)
			outputs.append(output)
		return outputs

	def answer_yn(self, question, context):
		"""Answers a YES/NO question. Outputs the probability that the answer is YES. """
		if context and len(context) > 1: # Capitalize the first letter of the premise
			context = context[0].upper() + context[1:]
		question = question[0].upper() + question[1:]
		if question[-1] != '?':
			question = question+'?'

		input_tensor = self.yn_tokenizer.encode_plus(question, context, return_tensors="pt").to('cuda:' + str(self.gpu_devices[0]))
		classification_logits = self.yn_qa_model(**input_tensor)[0]
		probs = torch.softmax(classification_logits, dim=1).tolist()
		if self.YN_idk: # class0:Yes, class1:No, class2:IDK
			yes_prob = probs[0][0]
		else: # class0:No, class1:Yes
			yes_prob = probs[0][1]
		return yes_prob

	def answer_ex(self, question, context_tokens):
		"""Answers an extractive question."""

		if context_tokens and len(context_tokens) > 1:  # Capitalize the first letter of the context
			context_tokens[0] = context_tokens[0][0].upper() + context_tokens[0][1:]
		question = question[0].upper() + question[1:]
		if question[-1] != '?':
			question = question + '?'

		question_tensor = self.ex_tokenizer.encode(question, return_tensors="pt")
		question_input_ids = question_tensor.tolist()[0]
		question_tokens = self.ex_tokenizer.convert_ids_to_tokens(question_input_ids)
		question_len = len(question_input_ids)

		context_tensor, context_bert_tokens, goldid_2_bertid, bertid_2_goldid = gold_to_bert_tokens(self.ex_tokenizer, context_tokens)

		input_tensor = torch.cat((question_tensor, context_tensor), 1).to('cuda:' + str(self.gpu_devices[0]))
		input_ids = input_tensor.tolist()[0]
		bert_tokens = question_tokens + context_bert_tokens
		try:
			outputs = self.ex_qa_model(input_tensor)
		except RuntimeError: # TODO: Expected tensor for argument #1 'indices' to have scalar type Long; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
			print(question, question_tokens)
			print(question_tokens)
			print(input_tensor)
			return {'span': None,
		        'answer': None,
		        'answer_tokens': None,
		        'confidence': None,
		        }
		answer_start_scores = outputs[0]
		answer_end_scores = outputs[1]
		start_probs = torch.softmax(answer_start_scores, dim=1).tolist()[0]
		end_probs = torch.softmax(answer_end_scores, dim=1).tolist()[0]
		answer_start = torch.argmax(
			answer_start_scores).tolist()  # Get the most likely beginning of answer with the argmax of the score
		answer_end = torch.argmax(
			answer_end_scores).tolist()  # Get the most likely end of answer with the argmax of the score
		confidence = np.mean([start_probs[answer_start], end_probs[answer_end]])

		bert_span = (answer_start - question_len, answer_end - question_len)

		if bert_span[0] >= bert_span[1]:  # no answer
			gold_span = None
			answer = None
			answer_tokens = None
		else:
			try:
				gold_span = (bertid_2_goldid[bert_span[0]], bertid_2_goldid[bert_span[1]] + 1)
			except IndexError:
				print(bertid_2_goldid)
				print(context_bert_tokens)
				print(context_tokens)
				print(answer_start, answer_end)
				print(bert_span)
				gold_span, answer, answer_tokens = None, None, None
			answer_tokens = context_tokens[gold_span[0]:gold_span[1]]
			answer = ' '.join(answer_tokens)
		return {'span': gold_span,
		        'answer': answer,
		        'answer_tokens': answer_tokens,
		        'confidence': confidence,
		        }
