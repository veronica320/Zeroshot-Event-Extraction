#!/usr/bin/env bash
## Finetune an extractive QA model on QAMR
# nohup python train_quase.py --target qamr --model mbert --cuda 0,1 > logs/finetune0.log 2>&1 &
# nohup python train_quase.py --target qamr --model xlm-roberta --cuda 2,3 > logs/finetune2.log 2>&1 &
# nohup python train_quase.py --mode train_eval --target qamr --model bert --cuda 0,1 > logs/bert.log 2>&1 &
# nohup python train_quase.py --mode train_eval --target qamr --model bert --cuda 2,3 > logs/bert_test.log 2>&1 &
# nohup python train_quase.py --mode train_eval --target qamr --model xlm --cuda 1 > logs/xlm.log 2>&1 &
# nohup python train_quase.py --mode train_eval --target qamr --model mbert-c --cuda 2 > logs/mbert-c.log 2>&1 &
# nohup python train_quase.py --mode train_eval --target qamr --model xlm-roberta-l --cuda 1 > logs/xlm-roberta-l.log 2>&1 &
# nohup python train_quase.py --mode train_eval --target qamr --model elior_bert-lc_mnli --cuda 2,4 > logs/elior_bert-lc_mnli_qamr.log 2>&1 &

## Finetune an extractive QA model on QAMR + SQuAD 2.0
#nohup python train_quase.py --mode train_eval --target qamr-squad2 --model bert --cuda 0 > logs/bert_qamr+squad2.log 2>&1 &
#nohup python train_quase.py --mode train_eval --target qamr-squad2 --model roberta --cuda 1 > logs/roberta_qamr+squad2.log 2>&1 &
#nohup python train_quase.py --mode train_eval --target qamr-squad2 --model bert-l --cuda 2 > logs/bert-l_qamr+squad2.log 2>&1 &
#nohup python train_quase.py --mode train_eval --target qamr-squad2 --model roberta-l --cuda 3 > logs/roberta-l_qamr+squad2.log 2>&1 &
#nohup python train_quase.py --mode train_eval --target qamr-squad2 --model elior_bert-lc_mnli --cuda 0,1 > logs/elior_bert-lc_mnli_qamr+squad2.log 2>&1 &

## Finetune a Y/N model on BoolQ (without IDK)
# nohup python train_yn.py --mode train_eval --target MRPC --model bert --cuda 2 > logs/MPRC.log 2>&1 &
# nohup python train_yn.py --mode train_eval --task WNLI --target boolq --model bert --cuda 5 > logs/boolq_bert.log 2>&1 &
# nohup python train_yn.py --mode train_eval --task WNLI --target boolq --model roberta --cuda 3 > logs/boolq_roberta.log 2>&1 &
# nohup python train_yn.py --mode train_eval --task WNLI --target boolq --model bert-l --cuda 6 > logs/boolq_bertl.log 2>&1 &
# nohup python train_yn.py --mode train_eval --task WNLI --target boolq --model roberta-l --cuda 7 > logs/boolq_robertal.log 2>&1 &


## Finetune a Y/N QA model on BoolQ with IDK
# nohup python train_yn.py --mode train_eval --task RTE --target boolq_idk --model bert --cuda 0 > logs/boolq_idk_bert.log 2>&1 &
# nohup python train_yn.py --mode train_eval --task RTE --target boolq_idk --model bert-l --cuda 0 > logs/boolq_idk_bertl.log 2>&1 &
# nohup python train_yn.py --mode train_eval --task RTE --target boolq_idk --model roberta --cuda 6 > logs/boolq_idk_roberta.log 2>&1 &
# nohup python train_yn.py --mode train_eval --task RTE --target boolq_idk --model roberta-l --cuda 4 > logs/boolq_idk_robertal_test.log 2>&1 &


## Continue finetuning pretrained Y/N QA models (with IDK) on the annotation guideline
# nohup python train_yn.py --mode transfer --task RTE --target anno_gdl --source boolq_idk --model bert --cuda 0 > logs/anno_gdl_boolq_idk_bert.log 2>&1 &
# nohup python train_yn.py --mode transfer --task RTE --target anno_gdl --source boolq_idk --model bert-l --cuda 1 > logs/anno_gdl_boolq_idk_bert-l.log 2>&1 &
# nohup python train_yn.py --mode transfer --task RTE --target anno_gdl --source boolq_idk --model roberta --cuda 2 > logs/anno_gdl_boolq_idk_roberta.log 2>&1 &
# nohup python train_yn.py --mode transfer --task RTE --target anno_gdl --source boolq_idk --model roberta-l --cuda 4 > logs/anno_gdl_boolq_idk_roberta-l.log 2>&1 &


## Continue finetuning a pretrained TE model (roberta-large-mnli) on the annotation guidleine  
# nohup python train_te.py --mode train_eval --task WNLI --target gdl_te_pos_neg --model roberta-large-mnli --cuda 1 > logs/gdl_te_pos_neg_robertal.log 2>&1 &
# nohup python train_te.py --mode train_eval --task WNLI --target gdl_te_pos_only --model roberta-large-mnli --cuda 0 > logs/gdl_te_pos_only_robertal.log 2>&1 &