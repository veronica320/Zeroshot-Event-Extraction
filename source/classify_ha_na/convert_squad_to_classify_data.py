# Converts squad2.0 QA data to classification data.
import os
import json
import csv

root_path = ('/shared/lyuqing/probing_for_event')
os.chdir(root_path)

old_squad_dir = "data/squad2"
new_squad_dir = "data/squad2_cls"



for split in ['train', 'dev']:
	frn = f'{old_squad_dir}/{split}.json'
	fwn = f"{new_squad_dir}/{split}.tsv"
	id = 0
	with open(frn, 'r') as fr, open(fwn, 'w') as fw:
		writer = csv.writer(fw, delimiter="\t")
		json_obj = json.load(fr)
		writer.writerow(["idx", "sentence1", "sentence2", "label"])

		for item in json_obj['data']:

			for prg in item['paragraphs']:

				context = prg['context']
				context = context.replace("\n","")

				qas = prg['qas']
				for qa in qas:
					question = qa['question']

					# no-answer questions
					if qa['is_impossible'] == True:
						row = [id, context, question, 0]
						writer.writerow(row)

					# has-answer questions
					else:
						row = [id, context, question, 2]
						writer.writerow(row)
					id += 1

