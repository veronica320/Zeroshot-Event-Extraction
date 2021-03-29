## Run AllenNLP's verbSRL model ACE dev sentences 
# allennlp predict https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz /shared/lyuqing/probing_for_event/data/ACE_oneie/en/event_only/dev.event.json --output-file verbSRL_output.json --silent --cuda-device

## Run AllenNLP's verbSRL model on 1000 anchor sentences 
# allennlp predict https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz /shared/lyuqing/probing_for_event/data/srl_input/anchor_sentences_1000.json --output-file /shared/lyuqing/probing_for_event/data/srl_output/anchor_sentences_1000_verbSRL_output.json

## Run AllenNLP's verbSRL model on annotation guidline sentences 
# allennlp predict https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz /shared/lyuqing/probing_for_event/data/srl_input/gdl_sents.jsonl --output-file gdl_sents_verbSRL_output.json --silent --cuda-device 0

# Run Celine's nomSRL model on annotation guideline sentences
# . ./run_nom_pipeline.sh /shared/lyuqing/probing_for_event/data/srl_input/gdl_sents.jsonl /shared/lyuqing/probing_for_event/data/srl_output/gdl_sents_nomSRL_output.jsonl 