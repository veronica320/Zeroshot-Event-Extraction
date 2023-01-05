# import
import numpy as np
import pickle
import re
import os
import allennlp
from nltk import wordpunct_tokenize, pos_tag
from collections import OrderedDict
import json
from pprint import pprint
import ipdb
import sys
from utils import srl, lexicon, span_utils, postprocessing
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
from allennlp.predictors.predictor import Predictor


class EventDetector():
	"""The Event Extraction pipeline."""

	def __init__(self, config):
		"""Initialize the pipeline.
		:param config (Config): configuration settings
		"""

		## Config

		# devices
		if config.use_gpu and config.gpu_devices != -1:
			self.gpu_devices = [int(_) for _ in config.gpu_devices.split(",")]
		else:
			self.gpu_devices = None

		# dirs
		self.transformers_cache_dir = config.transformers_cache_dir
		input_dir = eval(config.input_dir)
		split = eval(config.split)
		if "ACE" in input_dir:
			dataset = "ACE"
		elif "ERE" in input_dir:
			dataset = "ERE"
		else:
			raise ValueError("Unknown dataset")
		input_file = f"{input_dir}/{split}.event.json"

		# eval settings
		self.setting = eval(config.setting)

		# TE-related config
		self.TE_model_name = config.TE_model
		self.TE_model_type = eval(config.TE_model_type)
		self.srl_consts = eval(config.srl_consts)
		self.trg_thresh = eval(config.trg_thresh)
		self.trg_probe_type = eval(config.trg_probe_type)

		# QA-related config
		self.QA_model_name = config.QA_model
		self.QA_model_type = eval(config.QA_model_type)
		self.arg_thresh = eval(config.arg_thresh)
		self.arg_probe_type = eval(config.arg_probe_type)
		self.identify_head = config.identify_head


		## Probes
		# Load trigger probes
		probe_dir = f'source/lexicon/probes/{dataset}'
		trg_probes_frn = f'{probe_dir}/trg_te_probes_{self.trg_probe_type}.txt'
		with open(trg_probes_frn, 'r') as fr:
			self.trg_probe_lexicon = lexicon.load_trg_probe_lexicon(fr)

		# Load argument probes and the SRL-to-ACE argument type mapping
		arg_probes_frn = f'{probe_dir}/arg_qa_probes_{self.arg_probe_type}.txt'
		with open(arg_probes_frn, 'r') as fr:
			self.arg_probe_lexicon = lexicon.load_arg_probe_lexicon(fr, self.arg_probe_type)
		with open('source/lexicon/arg_srl2ace.txt') as fr:
			self.arg_map = lexicon.load_arg_map(fr)


		## Event types
		self.trg_subtypes = self.trg_probe_lexicon.keys()


		## SRL-related
		# stopwords that will be exluded from SRL predicates as potential triggers
		self.stopwords = srl.load_stopwords()
		# cached SRL output
		self.verb_srl_dict, self.nom_srl_dict = srl.load_srl(input_file)

	def load_models(self):
		"""Load pretrained models.
		"""
		print('Loading constituency and dependency parser...')
		self.dependency_parser = Predictor.from_path(
			"https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")
		self.constituency_parser = Predictor.from_path(
			"https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")


		print(f'Loading TE model...')
		if self.gpu_devices:
			self.TE_model = AutoModelForSequenceClassification.from_pretrained(self.TE_model_name,
																			   cache_dir=self.transformers_cache_dir).to('cuda:0')
		else:
			self.TE_model = AutoModelForSequenceClassification.from_pretrained(self.TE_model_name,
																			   cache_dir=self.transformers_cache_dir)
		self.TE_tokenizer = AutoTokenizer.from_pretrained(self.TE_model_name, cache_dir=self.transformers_cache_dir)

		print('Loading QA model...')
		if self.gpu_devices:
			self.QA_model = AutoModelForQuestionAnswering.from_pretrained(self.QA_model_name,
																		  cache_dir=self.transformers_cache_dir).to('cuda:0')
		else:
			self.QA_model = AutoModelForQuestionAnswering.from_pretrained(self.QA_model_name,
																		  cache_dir=self.transformers_cache_dir)

		self.QA_tokenizer = AutoTokenizer.from_pretrained(self.QA_model_name, cache_dir=self.transformers_cache_dir)

	def predict(self, instance):
		"""Predict on a single instance.
		:param instance (Instance): a sentence instance
		"""

		# get SRL results for the current instance
		srl_id_results, text_pieces, trg_cands, srl2gold_maps = srl.get_srl_results(instance,
																						(self.verb_srl_dict, self.nom_srl_dict),
																						self.stopwords,
																						self.srl_consts)
		pred_events = []  # a list of predicted events

		# predict triggers
		pred_events = self.extract_triggers(instance, pred_events, srl_id_results, text_pieces, trg_cands)

		# predict arguments
		pred_events = self.extract_arguments(instance, pred_events, srl_id_results, text_pieces, trg_cands, srl2gold_maps)

		return pred_events

	def extract_triggers(self, instance, pred_events, srl_id_results, text_pieces, trg_cands):
		"""Extract triggers."""

		sent = instance.sentence
		tokens_gold = instance.tokens  # tokens from preprocessed dataset

		if self.setting == "gold_TI+TC":  # directly return gold trigger identification + classification results

			for event in instance.events:
				# directly copy the gold trigger
				gold_trg_res = {"event_type": event['event_type'],
								"trigger": event["trigger"].copy(),
								"arguments": []}
				pred_events.append(gold_trg_res)

				# Get the SRL id
				for event_id, event in enumerate(pred_events):
					trigger_text = event["trigger"]["text"]
					
					srl_id, _ = srl.get_srl_id_and_premise(sent, 
														   trigger_text, 
														   trg_cands,
														   text_pieces,
														   self.constituency_parser)

					pred_events[event_id]["trg_premise"] = None
					pred_events[event_id]['srl_id'] = srl_id

		elif self.setting == "gold_TI":  # do trigger classification only

			# Get gold trigger spans
			for event in instance.events:
				gold_trg_res = {"event_type": None,
								"trigger": event["trigger"].copy(),
								"arguments": []}
				pred_events.append(gold_trg_res)

			# Classify each trigger
			for event_id, event in enumerate(pred_events):
				trigger_text = event["trigger"]["text"]

				# Get the SRL id and premise
				srl_id, premise = srl.get_srl_id_and_premise(sent,
															 trigger_text,
															 trg_cands,
															 text_pieces,
															 self.constituency_parser)

				# if SRL text_piece is None, use the entire sentence as the premise
				if premise is None:
					premise = sent

				# Classify
				top_type, confidence = self.classify_a_trigger(premise, trigger_text)
				pred_events[event_id]["event_type"] = top_type
				pred_events[event_id]["trg_premise"] = premise
				pred_events[event_id]["trigger"]['confidence'] = confidence
				pred_events[event_id]['srl_id'] = srl_id

		else:  # trigger identification + classification
			for srl_id, text_piece in text_pieces.items():
				trigger_text = trg_cands[srl_id][1]
				premise = text_piece

				top_type, confidence = self.classify_a_trigger(premise, trigger_text)

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

	def extract_arguments(self, instance, pred_events, srl_id_results, arg_text_pieces, trg_cands, srl2gold_maps):
		"""Extract arguments."""

		sent = instance.sentence
		tokens_gold = instance.tokens
		verb_srl2gold, nom_srl2gold = srl2gold_maps

		for event_id, event in enumerate(pred_events):
			srl_id = event['srl_id']
			trigger_text = event['trigger']['text']
			event_type = event['event_type']

			# Get the context
			if srl_id:
				srl_result = srl_id_results[srl_id]
				srl_tokens = srl_result['words']
				text_piece_tokens = [srl_tokens[i] for i, tag in enumerate(srl_result['tags']) if tag != 'O']
				text_piece_token_ids = [i for i, tag in enumerate(srl_result['tags']) if tag != 'O']
				text_piece = ' '.join(text_piece_tokens)
				srl2gold = nom_srl2gold if srl_result['predicate_type'] == 'nom' else verb_srl2gold
				context_tokens = text_piece_tokens
			else:
				context_tokens = tokens_gold

			event['arg_context'] = ' '.join(context_tokens)

			for cand_ace_arg in self.arg_probe_lexicon[event_type]:
				question = self.arg_probe_lexicon[event_type][cand_ace_arg]
				if self.arg_probe_type == 'ex_wtrg':
					question = question.replace('{trigger}', trigger_text)
				best_prediction = self.answer_ex(question, context_tokens)

				span = best_prediction['span']
				confidence = best_prediction["confidence"]

				if span and confidence >= self.arg_thresh:  # top answer is not None; confidence is high enough
					answer_tokens = best_prediction['answer_tokens']
					arg_text = ' '.join(answer_tokens)

					if self.identify_head:  # get the head
						pos_tags = [tag for _, tag in pos_tag(answer_tokens)]
						if answer_tokens:
							head_idx, head_token = span_utils.get_head(self.dependency_parser, span, answer_tokens, pos_tags)

							if head_idx:
								span = (head_idx, head_idx + 1)
								arg_text = head_token

					if srl_id:  # map context ids back to ACE gold ids
						start, end_pre = span[0], span[1] - 1
						start_in_srl, end_in_srl = text_piece_token_ids[start], text_piece_token_ids[end_pre] + 1
						try:
							gold_start, gold_end = srl2gold[start_in_srl], srl2gold[end_in_srl]
						except KeyError:
							continue # TODO: fix span mapping error
						span = (gold_start, gold_end)

					event['arguments'].append({'text': arg_text,
											   'role': cand_ace_arg[:-4],
											   'start': span[0],
											   'end': span[1],
											   'confidence': confidence,
											   })

				else:  # no answer; or confidence isn't high enough
					continue

		return pred_events

	def classify_a_trigger(self, premise, trigger_text):
		"""Classify a single trigger."""

		# the temporary result dict for all possible event types. Format: {event_type:confidence_score, ...}
		result_dict = {}

		for event_type in self.trg_subtypes:
			label = self.trg_probe_lexicon[event_type]  # the (partial) probe from the lexicon

			# See config_README for explanation of each trg_probe_type
			if self.trg_probe_type == 'topical':
				hypothesis = f'This text is about {label}.'
			elif self.trg_probe_type in ['natural', 'exist']:
				hypothesis = label
			else:
				raise ValueError("Undefined trg_probe_type. Should be in ['topical', 'natural', 'exist'].")

			# original probability of "the premise entailing the hypothesis"
			orig_entail_prob = self.entailment(premise, hypothesis)

			# maximizing the original entailment prob
			result_dict[event_type] = orig_entail_prob

		sorted_res = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
		top_type, confidence = sorted_res[0][0], sorted_res[0][1]  # Get the top event type and its confidence score

		return top_type, confidence

	def classify_an_argument(self, arg, event_type, premise, cand_ace_args):
		"""Classify a single argument."""

	def entailment(self, premise, hypothesis, premise_upper=True):
		"""Compute the probability that the premise entails the hypothesis."""

		if premise_upper and len(premise) > 1:  # Capitalize the first letter of the premise
			premise = premise[0].upper() + premise[1:]
		hypothesis = hypothesis[0].upper() + hypothesis[1:]

		x = self.TE_tokenizer.encode(premise, hypothesis, return_tensors='pt', truncation='only_first',
									 max_length=self.TE_tokenizer.model_max_length).to('cuda:0')
		logits = self.TE_model(x)[0]

		# these models have label=2 as the entailment class
		if self.TE_model_type in ['roberta', 'robertal', 'bartl']:
			entail_idx = 2

		# these models have label=1 as the entailment class
		elif self.TE_model_type in ['bert', 'bertl']:
			entail_idx = 1

		else:
			raise ValueError("Unrecognized TE_model_type. Should be in ['bert', 'bertl', 'roberta', 'robertal', 'bartl'].")

		probs = logits.softmax(1)
		entail_prob = float(probs[:, entail_idx])

		return entail_prob

	def answer_ex(self, question, context_tokens):
		"""Answers an extractive question."""

		# Capitalize the first letter
		if context_tokens and len(context_tokens) > 1:
			context_tokens[0] = context_tokens[0][0].upper() + context_tokens[0][1:]
		question = question[0].upper() + question[1:]
		if question[-1] != '?':
			question = question + '?'

		# Encode the question and context separately
		question_tensor = self.QA_tokenizer.encode(question, return_tensors="pt")
		question_input_ids = question_tensor.tolist()[0]
		question_tokens = self.QA_tokenizer.convert_ids_to_tokens(question_input_ids)
		question_len = len(question_input_ids)

		context_tensor, context_bert_tokens, goldid_2_bertid, bertid_2_goldid = span_utils.gold_to_bert_tokens(self.QA_tokenizer, context_tokens, self.QA_model_type)
		context_len = len(context_bert_tokens)

		if self.gpu_devices:
			input_tensor = torch.cat((question_tensor, context_tensor), 1).to('cuda:0')
		else:
			input_tensor = torch.cat((question_tensor, context_tensor), 1)
		input_ids = input_tensor.tolist()[0]
		bert_tokens = question_tokens + context_bert_tokens

		# Deal with BERT-based models and other models separately
		if self.QA_model_type in span_utils.bert_type_models:
			token_type_ids = torch.tensor([0] * question_len + [1] * context_len)
			if self.gpu_devices:
				token_type_ids = torch.unsqueeze(token_type_ids, 0).to('cuda:0')

				attention_mask = torch.tensor([1] * (question_len + context_len)).to('cuda:0')
				attention_mask = torch.unsqueeze(attention_mask, 0).to('cuda:0')
			else:
				token_type_ids = torch.unsqueeze(token_type_ids, 0)

				attention_mask = torch.tensor([1] * (question_len + context_len))
				attention_mask = torch.unsqueeze(attention_mask, 0)

			input_dict = {'input_ids': input_tensor,
			              'token_type_ids': token_type_ids,
			              'attention_mask': attention_mask}
			outputs = self.QA_model(**input_dict)
		else:
			outputs = self.QA_model(input_tensor)

		# Get the top k answers with post-processing function
		start_logits = outputs[0].cpu().detach().numpy()[0]
		end_logits = outputs[1].cpu().detach().numpy()[0]
		predictions, null_prediction = postprocessing.postprocess_qa_predictions(input_ids=input_ids,
		                                                          predictions=(start_logits, end_logits),
		                                                          question_len = question_len,
		                                                          version_2_with_negative=True,
		                                                          max_answer_length=20,
		                                                          null_score_diff_threshold=0.0
		                                                          )

		# reformat
		null_prediction = {'span': None,
					        'answer': None,
					        'answer_tokens': None,
					        'confidence': null_prediction["confidence"],
					        "start_logit": null_prediction["start_logit"],
					        "end_logit": null_prediction["end_logit"],
					        }

		## Match the predicted spans to texts
		final_predictions = []
		for pred in predictions:
			final_pred = span_utils.match_bert_span_to_text(pred, bertid_2_goldid, question_len, context_tokens)
			if final_pred is not None:  # valid prediction
				final_predictions.append(final_pred)


		# Pick the best prediction. If the null answer is not possible, this is easy.
		best_prediction = None
		best_non_null_pred = None

		# Otherwise we first need to find the best non-empty prediction.
		for i, pred in enumerate(final_predictions):
			if final_predictions[i]["answer"] is None:
				continue
			else:
				best_non_null_pred = final_predictions[i]
				break

		if best_non_null_pred is None: # we don't have any non-null prediction
			best_prediction = null_prediction
		else:
			# Then we compare to the null prediction using the threshold.
			score_diff = null_prediction["start_logit"] + null_prediction["end_logit"] - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
			if score_diff > 0.0:
				best_prediction = null_prediction
			else:
				best_prediction = best_non_null_pred

		return best_prediction

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

