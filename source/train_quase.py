#!/usr/bin/env python
# coding: utf-8

import transformers
import os 
import sys
import argparse

os.environ['PYTORCH_TRANSFORMERS_CACHE'] = '/shared/.cache/transformers_backup'
os.chdir('/shared/lyuqing/probing_for_event/')

parser = argparse.ArgumentParser(description='Process finetune config.')
parser.add_argument("--cuda",
				default=None,
				type=str,
				required=True,
				help="The GPU indices to use.",
		)
parser.add_argument("--mode",
				default=None,
				type=str,
				required=True,
				help="train_eval | eval.",
		)
parser.add_argument("--target",
				default=None,
				type=str,
				required=True,
				help="Target data to tune on.",
		)
parser.add_argument("--model",
				default='bert',
				type=str,
				required=False,
				help="Model architecture to use.",
		)
parser.add_argument("--epochs",
				default='3',
				type=str,
				required=False,
				help="Number of epochs.",
		)

args = parser.parse_args()

model_type = {
	'roberta': 'roberta',
	'roberta-l': 'roberta',
	'bert': 'bert',
	'bert-l': 'bert',
	'mbert': 'bert',
	'mbert-c': 'bert',
	'xlnet': 'xlnet',
	'xlm-roberta': 'xlm-roberta',
	'xlm-roberta-l': 'xlm-roberta',
	'xlm': 'xlm',	
}
		
model_name = {
	'roberta': 'roberta-base',
	'roberta-l': 'roberta-large',
	'bert': 'bert-base-uncased',
	'bert-l': 'bert-large-uncased',
	'mbert': 'bert-base-multilingual-uncased',
	'mbert-c': 'bert-base-multilingual-cased',
	'xlnet': 'xlnet-base-cased',
	'xlm-roberta': 'xlm-roberta-base',
	'xlm-roberta-l': 'xlm-roberta-large',
	'xlm': 'xlm-mlm-100-1280',
}

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
os.environ['DATA_DIR'] = f'data/{args.target}'
os.environ['TRANS_DIR'] = 'transformers/examples/question-answering'
os.environ['OUT_DIR'] = f'output_dir/{args.target}_{args.model}'

if args.mode == 'train_eval':
		os.system(f'\
				python $TRANS_DIR/run_squad.py \
			--model_type {model_type[args.model]} \
			--model_name_or_path {model_name[args.model]} \
			--do_train \
			--do_eval \
			--do_lower_case \
			--data_dir $DATA_DIR/ \
			--train_file train.json \
			--predict_file dev.json \
			--per_gpu_eval_batch_size 12 \
			--learning_rate 3e-5 \
			--num_train_epochs {args.epochs} \
			--max_seq_length 80 \
			--doc_stride 40 \
			--save_steps 50000 \
			--output_dir $OUT_DIR \
			--overwrite_output \
			--fp16 \
			')
elif args.mode == 'eval':
		os.system('\
				python $TRANS_DIR/run_squad.py \
			--model_type {} \
			--model_name_or_path {} \
			--do_eval \
			--do_lower_case \
			--data_dir $DATA_DIR/ \
			--predict_file wiki.dev.json \
			--per_gpu_eval_batch_size 12 \
			--max_seq_length 80 \
			--doc_stride 40 \
			--output_dir $OUT_DIR \
			'.format(model_type[args.model], model_name[args.model]))		
