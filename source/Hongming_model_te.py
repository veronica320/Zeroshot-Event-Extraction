# import
import numpy as np
import pickle
import re
import os
import allennlp
from nltk import wordpunct_tokenize, pos_tag
from allennlp.predictors.predictor import Predictor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import OrderedDict
import json
from pprint import pprint
import ipdb
import sys
from utils import *

# repo root dir
os.chdir('/shared/lyuqing/probing_for_event/')

# model abbreviation to full name mapping
model_abbr_dict = {'xlmr_xnli': 'joeddav/xlm-roberta-large-xnli',
                   'bartl': 'facebook/bart-large-mnli',
                   'roberta': 'textattack/roberta-base-MNLI',
                   'robertal': 'roberta-large-mnli',
                   'bert': 'textattack/bert-base-uncased-MNLI',
                   'distbert': 'textattack/distilbert-base-uncased-MNLI',
                   'xlnet': 'textattack/xlnet-base-cased-MNLI',
                   }

# hypothesis_dict = dict()
# hypothesis_dict['BE-BORN'] = ["someone's birth"]
# hypothesis_dict['MARRY'] = ["someone's marriage"]
# hypothesis_dict['DIVORCE'] = ["someone's divorce"]
# hypothesis_dict['INJURE'] = ["someone's injury"]
# hypothesis_dict['DIE'] = ["someone's death"]
# hypothesis_dict['TRANSPORT'] = ["someone or something moving from one place to another"]
# hypothesis_dict['TRANSFER-OWNERSHIP'] = ["a transfer of ownership"]
# hypothesis_dict['TRANSFER-MONEY'] = ["a transfer of money"]
# hypothesis_dict['START-ORG'] = ["starting an organization"]
# hypothesis_dict['MERGE-ORG'] = ["merging an organization"]
# hypothesis_dict['DECLARE-BANKRUPTCY'] = ["a bankruptcy"]
# hypothesis_dict['END-ORG'] = ["the end of an organization"]
# hypothesis_dict['ATTACK'] = ["an attack or a war"]
# hypothesis_dict['DEMONSTRATE'] = ["a demonstration"]
# hypothesis_dict['MEET'] = ["a meeting"]
# hypothesis_dict['PHONE-WRITE'] = ["a telephone or written communication with someone"]
# hypothesis_dict['START-POSITION'] = ["someone being hired"]
# hypothesis_dict['END-POSITION'] = ["someone no longer working on a position"]
# hypothesis_dict['NOMINATE'] = ["a nomination"]
# hypothesis_dict['ELECT'] = ["an election"]
# hypothesis_dict['ARREST-JAIL'] = ["an arrest", "capturing"]
# hypothesis_dict['RELEASE-PAROLE'] = ["a release or parole"]
# hypothesis_dict['TRIAL-HEARING'] = ["a trial or hearing"]
# hypothesis_dict['CHARGE-INDICT'] = ["a charge or indictment"]
# hypothesis_dict['SUE'] = ["a suit"]
# hypothesis_dict['CONVICT'] = ["a conviction"]
# hypothesis_dict['SENTENCE'] = ["a sentencing"]
# hypothesis_dict['FINE'] = ["a fine"]
# hypothesis_dict['EXECUTE'] = ["an execution of a criminal"]
# hypothesis_dict['EXTRADITE'] = ["an extradition"]
# hypothesis_dict['ACQUIT'] = ["an acquittal"]
# hypothesis_dict['PARDON'] = ["someone being pardoned"]
# hypothesis_dict['APPEAL'] = ["making an appeal"]

hypothesis_dict = dict()
hypothesis_dict['BE-BORN'] = ["birth", "someone's childbirth"]
hypothesis_dict['MARRY'] = ["marriage", "someone's wedding", "wedding", "marrying"]
hypothesis_dict['DIVORCE'] = ["divorce", "divorcing"]
hypothesis_dict['INJURE'] = ["injury", "hurting", "wounding", "injuring"]
hypothesis_dict['DIE'] = ["death", "dying"]
hypothesis_dict['TRANSPORT'] = ["moving", "a journey", "a trip", "visiting", "coming", "going", "arriving"]
hypothesis_dict['TRANSFER-OWNERSHIP'] = ["buying", "selling", "seizing", "acquiring"]
hypothesis_dict['TRANSFER-MONEY'] = ["paying", "donation", "loan"]
hypothesis_dict['START-ORG'] = ["starting an organization", "creating an organization", "founding an organization"]
hypothesis_dict['MERGE-ORG'] = ["merging an organization"]
hypothesis_dict['DECLARE-BANKRUPTCY'] = ["a bankruptcy", "bankrupting"]
hypothesis_dict['END-ORG'] = ["the end of an organization", "closing an organization", "the cease of an organization", "the crumble of an organization"]
hypothesis_dict['ATTACK'] = ["an attack or a war", "a fight", "attacking", "fighting"]
hypothesis_dict['DEMONSTRATE'] = ["a demonstration", "a rally", "a protest"]
hypothesis_dict['MEET'] = ["a meeting", "a summit", "meeting"]
hypothesis_dict['PHONE-WRITE'] = ["calling", "writing"]
hypothesis_dict['START-POSITION'] = ["an appointment", "appointing", "hiring"]
hypothesis_dict['END-POSITION'] = ["resignation", "retirement"]
hypothesis_dict['NOMINATE'] = ["a nomination", "nominating"]
hypothesis_dict['ELECT'] = ["an election", "electing", "voting"]
hypothesis_dict['ARREST-JAIL'] = ["an arrest", "capturing", "putting into jail"]
hypothesis_dict['RELEASE-PAROLE'] = ["a release or parole", "freeing someone"]
hypothesis_dict['TRIAL-HEARING'] = ["a trial or hearing"]
hypothesis_dict['CHARGE-INDICT'] = ["a charge or indictment", "accusing"]
hypothesis_dict['SUE'] = ["a suit", "a lawsuit"]
hypothesis_dict['CONVICT'] = ["a conviction"]
hypothesis_dict['SENTENCE'] = ["a sentencing"]
hypothesis_dict['FINE'] = ["a fine"]
hypothesis_dict['EXECUTE'] = ["an execution of a criminal", "hanging"]
hypothesis_dict['EXTRADITE'] = ["an extradition"]
hypothesis_dict['ACQUIT'] = ["an acquittal"]
hypothesis_dict['PARDON'] = ["someone being pardoned"]
hypothesis_dict['APPEAL'] = ["making an appeal"]




class EventDetectorTE():
    """The Textual-Entailment-based event extraction pipeline."""

    def __init__(self,
                 config,
                 ):

        # Config
        self.cache_dir = config.bert_cache_dir
        self.bert_model_type = config.bert_model_type
        self.te_model_name = model_abbr_dict[config.bert_model_type]
        if config.use_gpu and config.gpu_devices != -1:
            self.gpu_devices = [int(_) for _ in config.gpu_devices.split(",")]
        else:
            self.gpu_devices = None
        self.classification_only = config.classification_only

        self.srl_args = eval(config.srl_args)
        self.trg_thresh = eval(config.trg_thresh)
        self.arg_thresh = eval(config.arg_thresh)
        self.predicate_type = eval(config.predicate_type)
        self.add_neutral = config.add_neutral

        self.trg_probe_type = eval(config.trg_probe_type)
        self.pair_premise_strategy = eval(config.pair_premise_strategy)
        self.const_premise = eval(config.const_premise)
        self.tune_on_gdl = eval(config.tune_on_gdl)

        self.arg_probe_type = eval(config.arg_probe_type)
        self.identify_head = config.identify_head

        # Load trigger probes
        probe_dir = 'source/lexicon/probes/'
        if self.trg_probe_type in ['topical', 'prem-trg+type']:
            trg_probes_frn = f'{probe_dir}trg_te_probes_topical.txt'
        elif self.trg_probe_type == 'natural':
            trg_probes_frn = f'{probe_dir}trg_te_probes_natural.txt'
        elif self.trg_probe_type == 'exist':
            trg_probes_frn = f'{probe_dir}trg_te_probes_exist.txt'
        with open(trg_probes_frn, 'r') as fr:
            self.trg_probe_lexicon = load_trg_probe_lexicon(fr)

        # Load argument probes and the SRL-to-ACE argument map
        if self.arg_probe_type.startswith('auto'):
            arg_probes_frn = f'{probe_dir}arg_te_probes_auto.txt'
        elif self.arg_probe_type == 'manual':
            arg_probes_frn = f'{probe_dir}arg_te_probes_manual.txt'
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

        if self.tune_on_gdl:  # load the TE model tuned on the annotation guideline
            te_model_path = f'output_model_dir/gdl_te_{self.tune_on_gdl}_{self.te_model_name}'
        else:
            te_model_path = self.te_model_name

        print(f'Loading Textual Entailment model...')
        if self.gpu_devices:
            self.te_model = AutoModelForSequenceClassification.from_pretrained(te_model_path, cache_dir=self.cache_dir
                                                                               ).to('cuda:' + str(self.gpu_devices[0]))
        else:
            self.te_model = AutoModelForSequenceClassification.from_pretrained(te_model_path, cache_dir=self.cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.te_model_name, cache_dir=self.cache_dir)

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

        # predict arguments
        pred_events = self.extract_arguments(instance, pred_events, srl_id_results, text_pieces, trg_cands,
                                             srl2gold_maps)

        return pred_events

    def extract_triggers(self, instance, pred_events, srl_id_results, text_pieces, trg_cands):
        """Extract triggers."""

        sent = instance.sentence
        tokens_gold = instance.tokens  # ACE tokens

        if self.classification_only:  # classification only

            # Get the gold identified triggers and arguments
            for event in instance.events:
                gold_identified_res = {"event_type": None,
                                       "trigger": event["trigger"].copy(),
                                       "arguments": [
                                           {"text": arg["text"], "role": None, "start": arg["start"],
                                            "end": arg["end"]}
                                           for arg in event["arguments"]]}
                pred_events.append(gold_identified_res)

            for event_id, event in enumerate(pred_events):  # Classify each gold trigger
                trigger_text = event["trigger"]["text"]

                # Get the premise
                srl_id, text_piece = None, None
                for id, cand in trg_cands.items():
                    if trigger_text in cand[1] or cand[
                        1] in trigger_text:  # if SRL predicate overlaps with the gold trigger
                        text_piece = text_pieces[id]  # Use the srl text piece as the premise
                        srl_id = id
                if text_piece == None:  # if the gold trigger isn't in SRL prediates
                    if self.const_premise == 'whenNone':  # use the lowest constituent as the premise
                        text_piece = find_lowest_constituent(self.constituency_parser, trigger_text, sent)
                if self.const_premise == 'alwaystrg':  # regardless of whether the gold trigger is in SRL predicates, always use the lowest constituent as the premise
                    text_piece = find_lowest_constituent(self.constituency_parser, trigger_text, sent)

                premise = text_piece if text_piece else sent  # if text_piece is None, use the entire sentence as the premise

                top_type, confidence = self.classify_a_trigger(premise, trigger_text)

                pred_events[event_id]["event_type"] = top_type
                pred_events[event_id]["text_piece"] = text_piece
                pred_events[event_id]["trigger"]['confidence'] = confidence
                pred_events[event_id]['srl_id'] = srl_id

        else:  # identification + classification
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

    def extract_arguments(self, instance, pred_events, srl_id_results, text_pieces, trg_cands, srl2gold_maps):
        """Extract arguments."""

        sent = instance.sentence
        tokens_gold = instance.tokens  # ACE tokens
        verb_srl2gold, nom_srl2gold = srl2gold_maps

        if self.classification_only:
            for event_id, event in enumerate(pred_events):
                srl_id = event['srl_id']
                trigger_text = event['trigger']['text']
                event_type = event['event_type']

                # Get the premise
                text_piece = None
                if srl_id:  # if the gold trigger is in the SRL predicates
                    srl_result = srl_id_results[srl_id]
                    srl_tokens = srl_result['words']
                    text_piece = ' '.join([srl_tokens[i] for i, tag in enumerate(srl_result['tags']) if
                                           tag != 'O'])  # Concatenate all SRL arguments as the premise
                premise = text_piece if text_piece else sent

                # Classify each argument
                cand_ace_args = self.arg_probe_lexicon[
                    event_type]  # Take all ACE argument types of the current event type as candidates
                for arg_id, arg in enumerate(event["arguments"]):
                    top_arg_name, top_arg_score = self.classify_an_argument(arg, event_type, premise, cand_ace_args)
                    event["arguments"][arg_id]['role'] = top_arg_name
                    event["arguments"][arg_id]['confidence'] = top_arg_score

        else:
            for event_id, event in enumerate(pred_events):
                # Get the premise
                srl_id = event['srl_id']
                trigger_text = event['trigger']['text']
                event_type = event['event_type']

                # Get the premise
                srl_result = srl_id_results[srl_id]
                srl_tokens = srl_result['words']
                text_piece = ' '.join([srl_tokens[i] for i, tag in enumerate(srl_result['tags']) if tag != 'O'])
                srl2gold = nom_srl2gold if srl_result['predicate_type'] == 'nom' else verb_srl2gold

                # Construct srl_arg_dict
                srl_arg_dict = {}  # The span and tokens of all SRL arguments. Format: {'ARG0': [(span, token),(span, token)], 'ARG1': [(span, token), ...], ...}
                tag_set = set([tag[2:] for tag in srl_result['tags'] if
                               tag not in ['O', 'B-V', 'I-V']])  # SRL argument tags: ARG0, ARG1, ARGM-TMP...
                for target_tag in tag_set:
                    span = [j for j, tag in enumerate(srl_result['tags']) if
                            tag[2:] == target_tag]  # TODO: multiple args for the same arg type
                    tokens = [word for i, word in enumerate(srl_tokens) if i in span]
                    if self.identify_head:  # only retain the head
                        pos_tags = [tag for _, tag in pos_tag(tokens)]
                        head_idx, token = get_head(self.dependency_parser, span, tokens, pos_tags)
                        span = [head_idx]
                    else:  # retain the whole SRL argument
                        token = ' '.join([word for i, word in enumerate(srl_tokens) if i in span])
                    if None not in span:
                        span = (srl2gold[span[0]], srl2gold[span[-1]] + 1)  # map SRL ids to gold ids
                        if target_tag not in srl_arg_dict:
                            srl_arg_dict[target_tag] = []
                        srl_arg_dict[target_tag].append((span, token))

                # Classify each SRL argument
                for srl_arg_type, srl_arg_ists in srl_arg_dict.items():
                    if srl_arg_type not in self.arg_map[event_type]:  # the SRL argument isn't a potential ACE argument
                        continue
                    cand_ace_args = self.arg_map[event_type][
                        srl_arg_type]  # Only take the ACE argument types in the SRL-to-ACE argument mapping as candidates
                    for srl_arg_ist in srl_arg_ists:  # an instance of SRL argument
                        top_arg_name, top_arg_score = self.classify_an_argument(srl_arg_ist, event_type, text_piece,
                                                                                cand_ace_args)
                        if top_arg_score >= self.arg_thresh:
                            event['arguments'].append({'text': srl_arg_ist[1],
                                                       'role': top_arg_name,
                                                       'start': srl_arg_ist[0][0],
                                                       'end': srl_arg_ist[0][1],
                                                       'confidence': top_arg_score,
                                                       })

        return pred_events

    def classify_a_trigger(self, premise, trigger_text):
        """Classify a single trigger."""
        print(premise)
        print('Trigger:', trigger_text)
        result_dict = {}  # the temporary result dict for all possible event types. Format: {event_type:confidence_score, ...}
        for event_type in self.trg_subtypes:
            # label = self.trg_probe_lexicon[event_type]  # the (partial) probe from the lexicon
            # # See config_README for explanation of each trg_probe_type
            # if self.trg_probe_type == 'topical':
            #     hypothesis = f'This text is about {label}.'
            # elif self.trg_probe_type == 'prem-trg+type':
            #     hypothesis = re.sub(pattern=trigger_text, string=premise, repl=label).strip()
            # elif self.trg_probe_type in ['natural', 'exist']:
            #     hypothesis = label
            # print(event_type)
            # print(hypothesis)
            # orig_entail_prob = self.entailment(premise,
            #                                    hypothesis)  # original probability of "the premise entailing the hypothesis"
            # # print(orig_entail_prob)
            # if self.pair_premise_strategy:  # use a minimal pair of premises
            #     sub_pattern = '\s?' + trigger_text + '\s?'
            #     truncated_premise = re.sub(pattern=sub_pattern, string=premise,
            #                                repl=' ').strip()  # the truncated premise is the original premise - the trigger
            #     truncated_entail_prob = self.entailment(truncated_premise,
            #                                             hypothesis)  # the probability of "the truncated premise entailing the hypothesis"
            #     delta = orig_entail_prob - truncated_entail_prob  # the difference
            #
            # if self.pair_premise_strategy == 'max_delta':
            #     result_dict[event_type] = delta  # maximizing delta
            # elif self.pair_premise_strategy == 'max_conf+delta':
            #     result_dict[
            #         event_type] = orig_entail_prob + delta  # maximizing the sum of the original entailment prob + delta
            # elif self.pair_premise_strategy == None:
            #     result_dict[event_type] = orig_entail_prob  # maximizing the original entailment prob

            tmp_hypothesis_list = hypothesis_dict[event_type]
            tmp_scores = list()

            for tmp_hypothesis in tmp_hypothesis_list:

                # print(trigger_text + ' in this sentence is a kind of ' + tmp_hypothesis)
                tmp_scores.append(self.entailment(premise, 'It describes ' + tmp_hypothesis))
            result_dict[event_type] = max(tmp_scores)  # maximizing the original entailment prob

        sorted_res = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
        top_type, confidence = sorted_res[0][0], sorted_res[0][1]  # Get the top event type and its confidence score

        return top_type, confidence

    def classify_an_argument(self, arg, event_type, premise, cand_ace_args):
        """Classify a single argument."""

        cand_scores = {cand: 0 for cand in cand_ace_args}
        arg_text = arg['text'] if 'text' in arg else arg[1]
        for cand_ace_arg in cand_ace_args:
            if cand_ace_arg not in self.arg_probe_lexicon[event_type]:  # TODO: check why cand_ace_arg can be None
                continue
            hypothesis = self.arg_probe_lexicon[event_type][cand_ace_arg]
            hypothesis = hypothesis.replace('{}', arg_text)
            confidence = self.entailment(premise, hypothesis)
            cand_scores[cand_ace_arg] = confidence

        sorted_cands = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
        top_cand = sorted_cands[0]
        top_arg_name, top_arg_score = top_cand[0][:-4], top_cand[1]

        return top_arg_name, top_arg_score

    def entailment(self, premise, hypothesis, premise_upper=True):
        """Compute the probability that the premise entails the hypothesis."""

        if premise_upper and len(premise) > 1:  # Capitalize the first letter of the premise
            premise = premise[0].upper() + premise[1:]
        hypothesis = hypothesis[0].upper() + hypothesis[1:]

        x = self.tokenizer.encode(premise, hypothesis, return_tensors='pt', truncation='only_first',
                                  max_length=self.tokenizer.max_len).to('cuda:' + str(self.gpu_devices[0]))
        logits = self.te_model(x)[0]

        if self.bert_model_type in ['roberta', 'robertal', 'bartl',
                                    'xlmr_xnli']:  # these models have label=2 as the entailment class
            entail_idx = 2
        elif self.bert_model_type in ['bert', 'distbert', 'xlnet']:  # these models have label=1 as the entailment class
            entail_idx = 1

        if self.add_neutral:  # take the neutral class into account when computing softmax
            entail_contradiction_logits = logits
        else:
            entail_contradiction_logits = logits[:, [0, entail_idx]]

        probs = entail_contradiction_logits.softmax(1)
        if self.add_neutral:
            entail_prob = float(probs[:, entail_idx])
        else:
            entail_prob = float(probs[:, -1])

        return entail_prob

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

