# Zero-shot Event Extraction

This is the code repository for ACL2021 paper: [Zero-shot Event Extraction via Transfer Learning: Challenges and Insights](https://aclanthology.org/2021.acl-short.42/).

## Getting Started

### Environment

- `environment.yml` specifies the conda environment needed running the code. You can create the environment using it according to [this guildeline](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).
- When installing transformers, make sure you **install it from source** and put it under the root directory of this repo. This is because we need the scripts under `transformers/examples/`. 

### Data
- ACE-2005 (LDC2006T06): Available from [LDC's release](https://catalog.ldc.upenn.edu/LDC2006T06).
- ERE (LDC2015E29): This corpus is part of the [DEFT program](https://www.darpa.mil/program/deep-exploration-and-filtering-of-text) and thus only availble to its participants. Please check your institution's LDC account for access. 

One you download the corpora, place them under `data/`, so the enti# Zero-shot Event Extraction

This is the code repository for ACL2021 paper: [Zero-shot Event Extraction via Transfer Learning: Challenges and Insights](https://aclanthology.org/2021.acl-short.42/).

## Getting Started

### Environment

- `environment.yml` specifies the conda environment needed running the code. You can create the environment using it according to [this guildeline](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).
- When installing transformers, make sure you **install it from source** and put it under the root directory of this repo. This is because we need the scripts under `transformers/examples/`. 

### Data
- ACE-2005 (LDC2006T06): Available from [LDC's release](https://catalog.ldc.upenn.edu/LDC2006T06).
- ERE (LDC2015E29): This corpus is part of the [DEFT program](https://www.darpa.mil/program/deep-exploration-and-filtering-of-text) and thus only availble to its participants. Please check your institution's LDC account for access. 

One you download the corpora, place them under `data/`, so the complete paths look like `data/LDC2006T06` and `data/LDC2015E29`.


### Pretrained Models
- Pretrained Textual Entailment (TE) model: [coming](https://huggingface.co/veronica320/TE-for-event-extraction)
- Pretrained Question Answering (QA) model: [coming](https://huggingface.co/veronica320/QA-for-event-extraction)

## Repository structure

* `data/`: The data directory.
    * `ACE_oneie/`: The English ACE data converted to the OneIE format. Specifically, `en/event_only/{train|dev|test}.event.json` are the 3 files for event data only.
    * `srl_input/`: The input sentences for SRL.
    * `srl_output/`: The output after runnign SRL. The event extraction model loads SRL outputs from this directory when making predictions.
    * `boolq/`: The BoolQ data (for training Yes/No QA models).
* `output_dir/`: The folder that saves the event extraction predictions.  
* `output_model_dir/`: The folder that saves various finetuned QA/TE models.  
* `source/`: The source codes.
	* `config/`: Model configuration files. A specific README is included within.
	* `lexicon/`: Stores various txt files.
		* `probes/`: The "probe" files. Each file contains a set of probes (i.e. hypothese templates / question templates). Each file name is in the format of `{arg|trg}_{te|qa}_probes_{setting}.txt`. The most recently developed (and the best performing) ones are `trg_te_probes_topical.txt` and `arg_te_probes_manual.txt`.
		* `anno_guideline_examples`: The example sentences from the annotation guideline. 
		* `arg_srl2ace.txt`: A one-to-many mapping from SRL argument names to ACE argument names.
	* `configuration.py`: The Configuration class. Adapted from OneIE.
	* `model_te.py`: The TE-based event extraction pipeline. See comments inside.
	* `model_qa.py`: The QA-based event extraction pipeline (under development).
	* `predict.py`: The code to make predictions with the event extraction pipeline.
	* `evaluate.py`: The code to evaluate model predictions against gold annotations.
	* `scorer.py`: The scorer called by `evaluate.py`. Adapted from OneIE.
	* `graph.py`: The Graph class. Adapted from OneIE.
	* `data.py`: The Dataset class. Adapted from OneIE.
	* `util.py`: Helper functions. Adapted from OneIE.
	* `entail.py`: Code for unit testing a TE model.
	* `train_{quase|te|yn}.py`: The code to finetune a pretrained extractive QA | TE | Yes/No QA model respectively. See `finetune.sh` for running instructions.
	* `finetune.sh`: The scripts to finetune models with `train_{quase|te|yn}.py`. See comments in the file.
	* `srl.sh`: The scripts to run verbSRL (AllenNLP) and nomSRL (Celine's). See comments in the file. For more details, refer to [Celine's repo](https://github.com/celine-lee/nominal-srl-allennlpv0.9.0).
* `transformers/`: The huggingface transformers repository cloned from source. This is for the purpose of finetuning QA/TE models.  
* `env/`: The virtual environment.  No need to worry about. 

## Usage

### Make Predictions & Evaluate the Model
#### 1. Set the configuration
Go to `source/config`, and create a `.json` config file. The existing `te_optimial.json` can be a reference; it has the best performing config for TE-based classification only setting.  
See `source/config/config_README.md` for details on each parameter.
#### 2. Make predictions
Open `source/predict.py`, and change the config file name in the `config_path` variable accordingly. Then, run `python predict.py`.   
The gold and predicted events will be printed to std output. A json version of the predicted events will also be saved to `output_dir/`, in the same format as the input file.
#### 3. Evaluate the predictions
Open `source/evaluate.py`, and change the config file name in the `config_path` variable accordingly.  Then, run `python evaluate.py`.  
The scores will be printed to std output.

### Change the Model
The files that you might want to look into are `model_te.py` (and `model_qa.py` in the future), the config file, the probe files. Others are pretty much fixed.



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


### Pretrained Models
- Pretrained Textual Entailment (TE) model:
- Pretrained Question Answering (QA) model:

## Repository structure

* `data/`: The data directory.
    * `ACE_oneie/`: The English ACE data converted to the OneIE format. Specifically, `en/event_only/{train|dev|test}.event.json` are the 3 files for event data only. This is likely the only thing in `data/` that you need to use.
    * `srl_input/`: The input sentences for SRL.
    * `srl_output/`: The output after runnign SRL. The event extraction model loads SRL outputs from this directory when making predictions.
    * `boolq/`: The BoolQ data (for training Yes/No QA models).
    * `gdl_te/`: The positive and negative examples in the TE format converted from annotation guideline example sentences. This is for the purpose of continue finetuning TE models on the annotation guideline.
    * 	`apex/`: The apex package. No need to worry about.
* `output_dir/`: The folder that saves the event extraction predictions.  
* `output_model_dir/`: The folder that saves various finetuned QA/TE models.  
* `source/`: The source codes.
	* `config/`: Model configuration files. A specific README is included within.
	* `lexicon/`: Stores various txt files.
		* `probes/`: The "probe" files. Each file contains a set of probes (i.e. hypothese templates / question templates). Each file name is in the format of `{arg|trg}_{te|qa}_probes_{setting}.txt`. The most recently developed (and the best performing) ones are `trg_te_probes_topical.txt` and `arg_te_probes_manual.txt`.
		* `anno_guideline_examples`: The example sentences from the annotation guideline. 
		* `arg_srl2ace.txt`: A one-to-many mapping from SRL argument names to ACE argument names.
	* `configuration.py`: The Configuration class. Adapted from OneIE.
	* `model_te.py`: The TE-based event extraction pipeline. See comments inside.
	* `model_qa.py`: The QA-based event extraction pipeline (under development).
	* `predict.py`: The code to make predictions with the event extraction pipeline.
	* `evaluate.py`: The code to evaluate model predictions against gold annotations.
	* `scorer.py`: The scorer called by `evaluate.py`. Adapted from OneIE.
	* `graph.py`: The Graph class. Adapted from OneIE.
	* `data.py`: The Dataset class. Adapted from OneIE.
	* `util.py`: Helper functions. Adapted from OneIE.
	* `entail.py`: Code for unit testing a TE model.
	* `train_{quase|te|yn}.py`: The code to finetune a pretrained extractive QA | TE | Yes/No QA model respectively. See `finetune.sh` for running instructions.
	* `finetune.sh`: The scripts to finetune models with `train_{quase|te|yn}.py`. See comments in the file.
	* `srl.sh`: The scripts to run verbSRL (AllenNLP) and nomSRL (Celine's). See comments in the file. For more details, refer to [Celine's repo](https://github.com/celine-lee/nominal-srl-allennlpv0.9.0).
	* `multilingual_te.ipynb`: The multilingual version of the event extraction pipeline under development.
	* `scratch.{ipynb|py}`: Code drafts. 
	* `logs`: The logs of finetuning TE/QA models.
	* `apex/`: The apex package. No need to worry about.
* `ncdu`: A [tool](https://dev.yorhel.nl/ncdu) for disk usage management.  
* `SRL/`: SRL model repositories cloned from Celine. Under it, `transformers-srl/` is the multilingual verbal+nominal SRL model (under development), and `nominal-srl-allennlpv9.0/` is the English nominal SRL model. Since I already generated SRL predictions on the ACE data, there's no need to worry about this folder at the moment.   
* `transformers/`: The huggingface transformers repository cloned from source. This is for the purpose of finetuning QA/TE models.  
* `env/`: The virtual environment.  No need to worry about. 
* `oneie/`: The cloned OneIE repository. No need to worry about.    
* `demo_archive/`: The demo for the QA/TE model. No need to worry about.  
* `emb_dir/`: An archived embedding-based trigger classification model. No need to worry about.  
* `ISfromQA/`: Hangfeng's QUASE repository. No need to worry about.  

## Usage

### Make Predictions & Evaluate the Model
#### 1. Set the configuration
Go to `source/config`, and create a `.json` config file. The existing `te_optimial.json` can be a reference; it has the best performing config for TE-based classification only setting.  
See `source/config/config_README.md` for details on each parameter.
#### 2. Make predictions
Open `source/predict.py`, and change the config file name in the `config_path` variable accordingly. Then, run `python predict.py`.   
The gold and predicted events will be printed to std output. A json version of the predicted events will also be saved to `output_dir/`, in the same format as the input file.
#### 3. Evaluate the predictions
Open `source/evaluate.py`, and change the config file name in the `config_path` variable accordingly.  Then, run `python evaluate.py`.  
The scores will be printed to std output.

### Change the Model
The files that you might want to look into are `model_te.py` (and `model_qa.py` in the future), the config file, the probe files. Others are pretty much fixed.



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.