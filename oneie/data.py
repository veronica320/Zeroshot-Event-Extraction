import copy
import itertools
import json
import torch
from torch.utils.data import Dataset
from collections import Counter, namedtuple, defaultdict
import sys
from graph import Graph
from util import read_ltf, read_txt, read_json, read_json_single

instance_fields = [
    'sent_id', 'tokens', 'pieces', 'piece_idxs', 'token_lens', 'attention_mask',
    'entity_label_idxs', 'trigger_label_idxs',
    'entity_type_idxs', 'event_type_idxs',
    'relation_type_idxs', 'role_type_idxs',
    'mention_type_idxs',
    'graph', 'entity_num', 'trigger_num'
]
instance_ldc_eval_fields = [
    'sent_id', 'tokens', 'token_ids', 'pieces', 'piece_idxs',
    'token_lens', 'attention_mask'
]
batch_fields = [
    'sent_ids', 'tokens', 'piece_idxs', 'token_lens', 'attention_masks',
    'entity_label_idxs', 'trigger_label_idxs',
    'entity_type_idxs', 'event_type_idxs', 'mention_type_idxs',
    'relation_type_idxs', 'role_type_idxs',
    'graphs', 'token_nums'
]
batch_ldc_eval_fields = [
    'sent_ids', 'token_ids', 'tokens', 'piece_idxs', 'token_lens', 'attention_masks', 'token_nums'
]
Instance = namedtuple('Instance', field_names=instance_fields,
                      defaults=[None] * len(instance_fields))
InstanceLdcEval = namedtuple('InstanceLdcEval',
                             field_names=instance_ldc_eval_fields,
                             defaults=[None] * len(instance_ldc_eval_fields))
Batch = namedtuple('Batch', field_names=batch_fields,
                   defaults=[None] * len(batch_fields))
BatchLdcEval = namedtuple('BatchLdcEval',
                          field_names=batch_ldc_eval_fields,
                          defaults=[None] * len(batch_ldc_eval_fields))
BatchEval = namedtuple('BatchEval', field_names=['sent_ids', 'piece_idxs',
                                                 'tokens', 'attention_masks',
                                                 'token_lens', 'token_nums'])


def get_trigger_labels(events, token_num):
    """Convert event mentions in a sentence to a trigger label sequence with the
    length of token_num.
    :param events (list): a list of event mentions.
    :param token_num (int): the number of tokens.
    :return: a sequence of BIO format labels.
    """
    labels = ['O'] * token_num
    for event in events:
        trigger = event['trigger']
        start, end = trigger['start'], trigger['end']
        event_type = event['event_type']
        labels[start] = 'B-{}'.format(event_type)
        for i in range(start + 1, end):
            labels[i] = 'I-{}'.format(event_type)
    return labels

def get_role_types(entities, events, id_map):
    if entities:
        labels = [['O'] * len(entities) for _ in range(len(events))]
        entity_idxs = {entity['id']: i for i, entity in enumerate(entities)}
        for event_idx, event in enumerate(events):
            for arg in event['arguments']:
                entity_id = arg['entity_id']
                entity_id = id_map.get(entity_id, entity_id)
                entity_idx = entity_idxs[entity_id]
                # if labels[event_idx][entity_idx] != 'O':
                #     print('Conflict argument role {} {} {}'.format(event['trigger']['text'], arg['text'], arg['role']))
                labels[event_idx][entity_idx] = arg['role']
    else:
        labels = [['O'] * len(entities) for _ in range(len(events))]
        entity_idxs = {entity['id']: i for i, entity in enumerate(entities)}
        for event_idx, event in enumerate(events):
            for arg in event['arguments']:
                entity_id = arg['entity_id']
                entity_id = id_map.get(entity_id, entity_id)
                entity_idx = entity_idxs[entity_id]
                # if labels[event_idx][entity_idx] != 'O':
                #     print('Conflict argument role {} {} {}'.format(event['trigger']['text'], arg['text'], arg['role']))
                labels[event_idx][entity_idx] = arg['role']
    return labels


def get_role_list(events, vocab):
#     entity_idxs = {entity['id']: i for i, entity in enumerate(entities)}
#     visited = [[0] * len(entities) for _ in range(len(events))]
#     role_list = []
#     for i, event in enumerate(events):
#         for arg in event['arguments']:
#             entity_idx = entity_idxs[id_map.get(
#                 arg['entity_id'], arg['entity_id'])]
#             if visited[i][entity_idx] == 0:
#                 role_list.append((i, entity_idx, vocab[arg['role']]))
#                 visited[i][entity_idx] = 1
#     role_list.sort(key=lambda x: (x[0], x[1]))
    role_list = []
    for i, event in enumerate(events):
        for arg in event['arguments']:
            role_list.append((i, arg['start'], arg['end'], vocab[arg['role']]))
    return role_list

class IEDataset(Dataset):
    def __init__(self, path, max_length=128, gpu=False, ignore_title=False):
        """
        :param path (str): path to the data file.
        :param max_length (int): max sentence length.
        :param gpu (bool): use GPU (default=False).
        :param ignore_title (bool): Ignore sentences that are titles (default=False).
        """
        self.path = path
        self.data = []
        self.gpu = gpu
        self.max_length = max_length
        self.ignore_title = ignore_title
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def event_type_set(self):
        type_set = set()
        for inst in self.data:
            for event in inst['event_mentions']:
                type_set.add(event['event_type'])
        return type_set

    @property
    def role_type_set(self):
        type_set = set()
        for inst in self.data:
            for event in inst['event_mentions']:
                for arg in event['arguments']:
                    type_set.add(arg['role'])
        return type_set

    def load_data(self):
        """Load data from file."""
        overlength_num = title_num = 0
        with open(self.path, 'r', encoding='utf-8') as r:
            for line in r:
                inst = json.loads(line)
                is_title = inst['sent_id'].endswith('-3') and inst['tokens'][-1] != '.'
                if self.ignore_title and is_title:
                    title_num += 1
                    continue
                self.data.append(inst)

        if title_num:
            print('Discarded {} titles'.format(title_num))
        print('Loaded {} instances from {}'.format(len(self), self.path))

    def numberize(self, vocabs):
        """Numberize word pieces, labels, etcs.
        :param tokenizer: Bert tokenizer.
        :param vocabs (dict): a dict of vocabularies.
        """
        event_type_stoi = vocabs['event_type']
        role_type_stoi = vocabs['role_type']
        trigger_label_stoi = vocabs['trigger_label']

        data = []
        for inst in self.data:
            tokens = inst['tokens']
            sent_id = inst['sent_id']
            events = inst['event_mentions']
            events.sort(key=lambda x: x['trigger']['start'])
            token_num = len(tokens)

            # Pad word pieces with special tokens
#             piece_idxs = tokenizer.encode(pieces,
#                                           add_special_tokens=True,
#                                           max_length=self.max_length)
#             pad_num = self.max_length - len(piece_idxs)
#             attn_mask = [1] * len(piece_idxs) + [0] * pad_num
#             piece_idxs = piece_idxs + [0] * pad_num


            # Trigger
            # - trigger_labels and trigger_label_idxs are used for identification
            # - event_types and event_type_idxs are used for classification
            # - trigger_list is used for graph representation
            
#             trigger_labels = get_trigger_labels(events, token_num)
#             trigger_label_idxs = [trigger_label_stoi[l]
#                                   for l in trigger_labels]
#             event_types = [e['event_type'] for e in events]
#             event_type_idxs = [event_type_stoi[l] for l in event_types]

            trigger_list = [(e['trigger']['start'], e['trigger']['end'],
                             event_type_stoi[e['event_type']])
                            for e in events]


#             #Argument role
#             role_types = get_role_types(entities, events, entity_id_map)
#             print(role_types)
#             role_type_idxs = [[role_type_stoi[l] for l in ls]
#                               for ls in role_types]

            role_list = get_role_list(events, role_type_stoi)

            # Graph
            graph = Graph(
                triggers=trigger_list,
                roles=role_list,
                vocabs=vocabs,
            )

            instance = Instance(
                sent_id=sent_id,
                tokens=tokens,
                graph=graph,
                trigger_num=len(events),
            )
            data.append(instance)
        self.data = data
