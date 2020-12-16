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
        help="train_eval | eval | transfer.",
    )
parser.add_argument("--target",
        default=None,
        type=str,
        required=True,
        help="Target data to tune on.",
    )
parser.add_argument("--source",
        default=None,
        type=str,
        required=False,
        help="Source model for transfer.",
    )
parser.add_argument("--task",
        default=None,
        type=str,
        required=True,
        help="Task format.",
    )
parser.add_argument("--model",
        default='bert',
        type=str,
        required=False,
        help="Model architecture to use.",
    )
parser.add_argument("--epochs",
        default='5',
        type=str,
        required=False,
        help="Number of epochs.",
    )

args = parser.parse_args()

model_type = {
  'roberta': 'roberta',
  'roberta-l': 'roberta',
  'bert': 'bert',
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
os.environ['TRANS_DIR'] = 'transformers/examples/text-classification'
os.environ['OUT_DIR'] = 'output_dir/{}_{}'.format(args.target, args.model)
if args.target == 'boolq':
    os.environ['DATA_DIR'] = 'data/boolq/tsv/'
if args.target == 'boolq_idk':
    os.environ['DATA_DIR'] = 'data/boolq/idk/'
if args.target == 'anno_gdl':
    os.environ['DATA_DIR'] = 'data/boolq/anno_gdl/'
else:
    os.environ['DATA_DIR'] = 'data/glue/'+args.task+'/'

if args.mode == 'transfer':
    os.environ['SOURCE_DIR'] = 'output_dir/{}_{}'.format(args.source, args.model)
    os.environ['OUT_DIR'] = 'output_dir/{}_{}_{}'.format(args.target, args.source, args.model)
    
    
if args.mode == 'train_eval':
    os.system(f'\
        python $TRANS_DIR/run_glue.py \
      --model_name_or_path {model_name[args.model} \
      --task_name {args.task} \
      --do_train \
      --do_eval \
      --data_dir $DATA_DIR \
      --per_gpu_train_batch_size 8 \
      --per_gpu_eval_batch_size 24 \
      --learning_rate 1e-5 \
      --num_train_epochs {args.epochs} \
      --max_seq_length 200 \
      --output_dir $OUT_DIR \
      --evaluate_during_training \
      --logging_steps 500 \
      --save_steps -1 \
      --fp16 \')

if args.mode == 'transfer':
    os.system('\
        python $TRANS_DIR/run_glue.py \
      --model_name_or_path $SOURCE_DIR \
      --task_name {} \
      --do_train \
      --do_eval \
      --data_dir $DATA_DIR \
      --per_gpu_train_batch_size 8 \
      --per_gpu_eval_batch_size 24 \
      --learning_rate 1e-5 \
      --num_train_epochs {} \
      --max_seq_length 200 \
      --output_dir $OUT_DIR \
      --evaluate_during_training \
      --logging_steps 500 \
      --save_steps 20000 \
      --fp16 \
      '.format(args.task, args.epochs))    

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
