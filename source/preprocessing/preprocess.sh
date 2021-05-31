#!/usr/bin/env bash


#python process_ace.py -i ../data/ACE-2005 -o ../data/ACE_oneie/zh/event_only -s resource/splits/ACE05-CN -b bert-base-multilingual-cased -c /shared/.cache/transformers -l chinese

#python process_ace.py -i ../data/ACE-2005/Extracted/data -o ../data/ACE_oneie/en/trial -s resource/splits/ACE05-E --time_and_val -b bert-base-multilingual-cased -c /shared/.cache/transformers -l english

#python process_ere.py -i ../data/ERE/ERE_raw/LDC2015E68/LDC2015E68_DEFT_Rich_ERE_English_Training_Annotation_R2_V2/data -o ../data/ERE/ERE_oneIE/LDC2015E68 -b bert-large-cased -c /shared/.cache/transformers -l english -d normal
#
# python process_ere.py -i ../data/ERE/ERE_raw/LDC2015E29/LDC2015E29_DEFT_Rich_ERE_English_Training_Annotation_V2/data -o ../data/ERE/ERE_oneIE/LDC2015E29 -b bert-large-cased -c /shared/.cache/transformers -l english -d normal


python process_ace.py -i ../data/ACE-2005/Extracted/data -o ../data/ACE_oneie/en/event_only_cleaned -s resource/splits/ACE05-E --time_and_val -b bert-base-multilingual-cased -c /shared/.cache/transformers -l english
