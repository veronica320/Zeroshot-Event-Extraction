This README explains the meaning of each parameter in the configuration file.

#### Paths & devices
* `use_gpu`: Whether to run on gpu.
* `gpu_devices`: The gpu device to use.
* `bert_cache_dir`: The cache dir for pretrained huggingface transformer models.
* `root_dir`: The root dir to the `Zero-shot-Event-Extraction` repo.
* `input_file`: The input ACE data file.

#### Evaluation setup
* `classification_only`: Whether to evaluate classification only (assuming gold identification).

#### General settings for triggers & arguments
* `bert_model_type`: The pretrained model architecture. Can be one of `xlmr_xnli`,`bartl`,`roberta`,`robertal`,`bert`,`distbert`,`xlnet`.
* `srl_args`: The SRL arguments that are included in the textpiece fed as the "premise" to the TE model (resp. the "context" to the QA model).
* `trg_thresh`: The minimum confidence score required for the pretrained model to output a SRL predicate as an identified event trigger.
* `arg_thresh`: The minimum onfidence score for the pretrained model to output a SRL argument as an identified event argument.
* `predicate_type`: What SRL predicates to include as potential trigger candidates. Defaults to `['verb','nom']`.  
* `add_neutral`: Whether to take into account the neutral class when computing the entailment probability.

#### Trigger-specific settings
* `trg_probes`: The type of trigger probes to use:
	* `topical`: The probes are in the form of `This is about {topic}.`, corresponding to the file `trg_te_probes_topical.txt` under `lexicon/probes/`. For example, `This is about someone's birth.`
	* `natural`: The probes are in a more natural form, like `{Someone did something}` or `{Something happened}`, corresponding to the file `trg_te_probes_natural.txt` under `lexicon/probes/ `. For example, `Someone is born.`
	* `exist`: The probes are in the form of `There is {something}.`, corresponding to the file `trg_te_probes_exist.txt` under `lexicon/probes/`. For example, `There is a birth.`
* `const_premise`: If True, when a trigger is not in the SRL predicates, instead of using the entire sentence as the premise, we use the "lowest constituent of a certain type" (NP, PP, or S). For example, if the triggerr "wounded" in the sentence "killing a wounded Iraqi" isn't covered by SRL, we would identify the NP constituent "a wounded Iraqi" as the premise, which excludes the noise from "killing". See details [here](https://docs.google.com/document/d/1jqS7LTekh1G8MHQ9P9AWy27afbGBPrrKlLJqryT6Rj4/edit#bookmark=id.ec387xm2h8tm).

#### Argument-specific settings
* `arg_probe_type`: The type of argument probes to use:
	* `manual`: The probe templates are manually written, corresponding to the file `arg_te_probes_manual.txt` under `lexicon/probes/`. For example, `{SRL A0} is born.`
	* `auto_issth`: The probe templates are automatically generated from the defition of arguments in ACE annotation guideline, corresponding to the file `arg_te_probes_auto.txt` under `lexicon/probes/`. For example, the definition of `Person_Arg` in `BE-BORN` is "The person who is born". Then the hypothesis will be "The person who is born is {SRL A0}."
	* `auto_sthis`: Same as `auto_issth`, except that the hypothesis will be in the form "{SRL A0} is the person who is born."
* `identify_head`: Whether to add head identification as a post-processing step, and only outputs the head.

