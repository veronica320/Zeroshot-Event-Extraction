#!/usr/bin/env bash


python process_ace.py -i ../data/ACE-2005 -o ../data/ACE_oneie/zh/event_only -s resource/splits/ACE05-CN -b bert-base-multilingual-cased -c /shared/.cache/transformers -l chinese