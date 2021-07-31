import copy
import itertools
import json
import torch
from torch.utils.data import Dataset
from collections import Counter, namedtuple, defaultdict
import sys
from graph import Graph

instance_fields = [
    'doc_id', 'sent_id', 'tokens', 'sentence', 'events',
    'graph', 'trigger_num'
]
Instance = namedtuple('Instance', field_names=instance_fields)


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
                labels[event_idx][entity_idx] = arg['role']
    else:
        labels = [['O'] * len(entities) for _ in range(len(events))]
        entity_idxs = {entity['id']: i for i, entity in enumerate(entities)}
        for event_idx, event in enumerate(events):
            for arg in event['arguments']:
                entity_id = arg['entity_id']
                entity_id = id_map.get(entity_id, entity_id)
                entity_idx = entity_idxs[entity_id]
                labels[event_idx][entity_idx] = arg['role']
    return labels


def get_role_list(events, vocab):
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

                # TODO: add back coarse type
                for event in inst['event_mentions']:
                    event_type = event['event_type']
                    if ':' in event_type:
                        event['event_type'] = event_type.split(':')[1].upper()
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
            sent = inst['sentence']
            doc_id = inst['doc_id']
            events.sort(key=lambda x: x['trigger']['start'])
            token_num = len(tokens)

            trigger_list = [(e['trigger']['start'], e['trigger']['end'],
                             event_type_stoi[e['event_type']])
                            for e in events]

            role_list = get_role_list(events, role_type_stoi)

            # Graph
            graph = Graph(
                triggers=trigger_list,
                roles=role_list,
                vocabs=vocabs,
            )

            instance = Instance(
                doc_id = doc_id,
                sent_id=sent_id,
                sentence = sent,
                tokens=tokens,
                graph=graph,
                events = events,
                trigger_num=len(events),
            )
            data.append(instance)
        self.data = data
