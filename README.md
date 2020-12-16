<!-- Template from https://github.com/othneildrew/Best-README-Template-->

# Zero-shot Event Extraction via Probing QA/TE Models

This is the repository for the ongoing work on a zero-shot event extraction pipeline via probing pretrained Question Answering/Textual Entailment models.



## Repository structure
The root directory is at `/shared/lyuqing/probing_for_event`.
There are several folders within:

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

## Prerequisites

All dependencies are available in the following virtual environment: 
```conda activate /shared/lyuqing/probing_for_event/env```

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