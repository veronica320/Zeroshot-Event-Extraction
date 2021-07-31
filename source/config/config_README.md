This document explains the meaning of each parameter in the configuration file.

#### Devices
* `use_gpu`: Whether to run on gpu.
* `gpu_devices`: The gpu device to use.

#### Paths
* `root_dir`: The root directory of the `Zeroshot-Event-Extraction` repo. You should change this accordingly.
* `transformers_cache_dir`: The cache dir for pretrained huggingface transformer models.
* `root_dir`: The root dir to the `Zero-shot-Event-Extraction` repo.
* `input_dir`: The input data directory. Should be one of `data/ACE_converted` of `data/ERE_converted`.
* `split`: If evaluated on ACE, the split should be `train`, `dev`, or `test`. If evaluated on ERE, the split should be `all`.

#### Task setting
* `setting`: The evaluation setting. Can be `scratch`, `gold_TI`, or `gold_TI+TC`. For further details, please refer to Section 3 of the paper. 


#### Trigger-related configs
* `TE_model`: The Textual Entailment model to be used in trigger extration. Default to `veronica320/TE-for-Event-Extraction`. You can change it to another TE model which is an instance of [AutoModelForSequenceClassification](https://huggingface.co/transformers/model_doc/auto.html#automodelforsequenceclassification).
* `TE_model_type`: The Transformers model type of the `TE_model`. Can be `bert`, `bertl`, `roberta`, `robertal`, `bartl`.
* `srl_consts`: The SRL constituents to include in the TE premise for trigger extraction. See Appendix C.2 in the paper for more details.
* `trg_thresh`: The minimum confidence score required for the pretrained model to output a SRL predicate as an identified event trigger. See Appendix C.2 in the paper for more details.
* `trg_probe_type`: The type of trigger probes to use. Each option corresponds to a file under `lexicon/probes/{dataset}`. See Appendix C.2 for more details.
	* `topical`: The probes are in the form of `This is about {topic}.`, corresponding to the file `trg_te_probes_topical.txt`.
	* `natural`: The probes are in a more natural form, like `{Someone did something}` or `{Something happened}`, corresponding to the file `trg_te_probes_natural.txt`.
	* `exist`: The probes are in the form of `There is {something}.`, corresponding to the file `trg_te_probes_exist.txt`.
	
#### Argument-related configs
* `QA_model`: The Question Answering model to be used in argument extration. Default to `veronica320/QA-for-Event-Extraction`. You can change it to another QA model which is an instance of [AutoModelForQuestionAnswering](https://huggingface.co/transformers/model_doc/auto.html#automodelforquestionanswering).
* `QA_model_type`: The Transformers model type of the `QA_model`. Can be `bert`, `bertl`, `roberta`, `robertal`, `bartl`.
* `arg_thresh`: The minimum onfidence score for the pretrained model to output a SRL argument as an identified event argument.
* `arg_probe_type`: The type of argument probes to use. Each option corresponds to a file under `lexicon/probes/{dataset}`. See Appendix C.2 for more details.
	* `static`: The probes are fixed for each event
type and argument type, e.g. `Where is the attack?`, corresonding to the file `arg_qa_probes_static.txt`.
	* `contexutalized `: : The probes are instantiated
with the trigger of event instances when possible, e.g. `Where is the {trigger}?`, corresponding to the file `arg_qa_probes_ contexutalized.txt`.
* `identify_head`: Whether to add head identification as a post-processing step, and only outputs the head.