#!/usr/bin/env python
# coding: utf-8

import transformers
import os 
import sys
import argparse

os.environ['PYTORCH_TRANSFORMERS_CACHE'] = '/shared/.cache/transformers'
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
parser.add_argument("--train_file",
				default=None,
				type=str,
				required=False,
				help="train_file name.",
		)
parser.add_argument("--pred_file",
				default=None,
				type=str,
				required=False,
				help="predict_file name.",
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
parser.add_argument("--null_score_diff_threshold",
				default=0.0,
				type=str,
				required=False,
				help="If null_score - best_non_null is greater than the threshold predict null.",
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
	'elior_bert-lc_mnli': 'bert',
	'elior_bert-lc_mnli_squad2': 'bert',
	'squad2_mbert': 'bert',
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
	'elior_bert-lc_mnli': 'output_model_dir/elior_bert-lc_mnli',
}

if args.model in model_name:
	os.environ['MODEL_NAME'] = model_name[args.model]
else:
	os.environ['MODEL_NAME'] = f'output_model_dir/{args.model}'



os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
os.environ['DATA_DIR'] = f'data/{args.target}'
os.environ['TRANS_DIR'] = 'transformers/examples/question-answering'
os.environ['OUT_DIR'] = f'output_model_dir/{args.target}_{args.model}'

if args.train_file:
	os.environ['TRAIN_FILE'] = args.train_file
else:
	os.environ['TRAIN_FILE'] = 'train.json'


if args.pred_file:
	os.environ['PRED_FILE'] = args.pred_file
else:
	os.environ['PRED_FILE'] = 'dev.json'

if args.mode == 'train_eval':
		os.system(f'\
				python $TRANS_DIR/run_squad.py \
			--model_type {model_type[args.model]} \
			--model_name_or_path $MODEL_NAME \
			--do_train \
			--do_eval \
			--do_lower_case \
			--data_dir $DATA_DIR/ \
			--train_file $TRAIN_FILE \
			--predict_file $PRED_FILE \
			--per_gpu_train_batch_size 48 \
			--per_gpu_eval_batch_size 12 \
			--learning_rate 3e-5 \
			--num_train_epochs {args.epochs} \
			--max_seq_length 80 \
			--doc_stride 40 \
			--save_steps 50000 \
			--output_dir $OUT_DIR \
			--overwrite_output \
			--fp16 \
			--version_2_with_negative \
			')
elif args.mode == 'eval':
		os.system(f'\
				python $TRANS_DIR/run_squad.py \
			--model_type {model_type[args.model]} \
			--model_name_or_path $MODEL_NAME \
			--do_eval \
			--do_lower_case \
			--data_dir $DATA_DIR/ \
			--predict_file $PRED_FILE \
			--per_gpu_eval_batch_size 48 \
			--max_seq_length 80 \
			--doc_stride 40 \
			--output_dir $OUT_DIR \
			--fp16 \
            --version_2_with_negative \
            --null_score_diff_threshold {args.null_score_diff_threshold}\
			')
