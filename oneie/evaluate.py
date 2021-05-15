import os
import json
from argparse import ArgumentParser
from data import IEDataset
from scorer import score_graphs
from util import generate_vocabs
import pprint

# configuration
parser = ArgumentParser()
parser.add_argument('-g', '--gold_file', help='path to the gold file.')
parser.add_argument('-p', '--pred_file', help='path to the pred file.')
args = parser.parse_args()


gold_dataset = IEDataset(args.gold_file)
pred_dataset = IEDataset(args.pred_file)

gold_graphs, pred_graphs = [], []

vocabs = generate_vocabs([gold_dataset, pred_dataset])

gold_dataset.numberize(vocabs)
pred_dataset.numberize(vocabs)

for inst1, inst2 in zip(gold_dataset, pred_dataset):
    gold_graphs.append(inst1.graph)
    pred_graphs.append(inst2.graph)    

scores = score_graphs(gold_graphs, pred_graphs)
print(scores)