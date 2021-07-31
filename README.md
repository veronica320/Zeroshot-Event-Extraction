# Zero-shot Event Extraction

This is the code repository for ACL2021 paper: [Zero-shot Event Extraction via Transfer Learning: Challenges and Insights](https://aclanthology.org/2021.acl-short.42/).

A lot of the infrastructure (preprocessing, scorer, etc.) is adapted from the [OneIE codebase](http://blender.cs.illinois.edu/software/oneie/). Special thanks to the authors!

## Getting Started

### Environment

- If you are a [CogComp](https://cogcomp.seas.upenn.edu/) member, you can directly run `/shared/lyuqing/probing_for_event/env` on any NLP server to activate the conda environment.
- Otherwise, `environment.yml` specifies the conda environment needed running the code. You can create the environment using it according to [this guildeline](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

### Data
- ACE-2005 (LDC2006T06): Available from [LDC's release](https://catalog.ldc.upenn.edu/LDC2006T06).
- ERE (LDC2015E29): This corpus is part of the [DEFT program](https://www.darpa.mil/program/deep-exploration-and-filtering-of-text) and thus only availble to its participants. Please check your institution's LDC account for access. 

One you download the corpora, place them under `data/`, so the complete paths look like `data/LDC2006T06` and `data/LDC2015E29`.

### Pretrained Models
- Pretrained Textual Entailment (TE) model: [link](https://huggingface.co/veronica320/TE-for-Event-Extraction)
- Pretrained Question Answering (QA) model: [link](https://huggingface.co/veronica320/QA-for-Event-Extraction)

No need to pre-download them yourself; once you follow the instructions to run the event extration pipeline, they will be downloaded automatically.

## Repository structure

* `data/`: The data directory.
    * `LDC2006T06/`: The ACE corpus.
    * `LDC2015E29/`: The ERE corpus.
    * `ACE_converted/`: The preprocessed ACE corpus. Specifically, `{train|dev|test}.event.json` will be the 3 files relevant to event extraction only.
    * `ERE_converted/`: The preprocessed ERE corpus. Specifically, `all.event.json` will be the file elevant to event extraction only.
    * `srl_output/`: The output of running SRL on the preprocessed ACE/ERE. The event extraction model loads SRL outputs from this directory when making predictions.
    * `splits`: The train/dev/test splits from OneIE.
* `output_dir/`: The folder for model predictions.  
* `source/`: The source codes.
	* `config/`: Model configuration files. A specific README is included within.
	* `prepreocessing/`: Preprocessing scripts for ACE and ERE data.
	* `lexicon/`: Stores various txt files.
		* `probes/`: The "probe" files. Each file contains a set of probes (i.e. hypothese templates / question templates). Each file name is in the format of `{arg|trg}_{te|qa}_probes_{setting}.txt`.
		* `arg_srl2ace.txt`: A one-to-many mapping from SRL argument names to ACE argument names.
	* `configuration.py`: The Configuration class. Adapted from OneIE.
	* `model.py`: The main event extraction pipeline. See comments inside.
	* `predict_evaluate.py`: The code to make predictions and evaluate the model.
	* `scorer.py`: The scorer called by `predict_evaluate.py`. Adapted from OneIE.
	* `graph.py`: The Graph class. Adapted from OneIE.
	* `data.py`: The Dataset class. Adapted from OneIE.
	* `utils/`: Helper functions.

## Usage
Here are the instructions on how to run our system for inference.
To start, go to the `source` folder.

### Data preprocessing
Our preprocessing scripts are adapted from the [OneIE codebase](http://blender.cs.illinois.edu/software/oneie/).

#### Preprocessing ACE2005
The `prepreocessing/process_ace.py` script converts raw ACE2005 datasets to the format used by our system. 

Usage:

```
python preprocessing/process_ace.py -i <INPUT_DIR> -o <OUTPUT_DIR> -s data/splits/ACE05-E -b <BERT_MODEL> -c <CACHE_DIR> -l <LANGUAGE> --time_and_val
```

Example:

```
python preprocessing/process_ace.py -i data/LDC2006T06/data -o data/ACE_converted -s data/splits/ACE05-E -b bert-large-cased -c /shared/.cache/transformers -l english --time_and_val
```

Arguments:

- -i, --input: Path to the input directory. Should be `data/LDC2006T06/data`.
- -o, --output: Path to the output directory. Should be `data/ACE_converted`.
- -b, --bert: Bert model name.
- -c, --cache\_dir: Path to your Huggingface Transformer cache directory.
- -s, --split: Path to the split directory. We use the same splits as OneIE.
- -l, --lang: Language (options: english, chinese).

#### Preprocessing ERE
The `prepreocessing/process_ere.py` script converts raw ERE dataset to the format used by our system.

Usage:

```
python preprocessing/process_ere.py -i <INPUT_DIR> -o <OUTPUT_DIR> -b <BERT_MODEL> -c <CACHE_DIR> -l <LANGUAGE>
```

Example:

```
python preprocessing/process_ere.py -i data/LDC2015E29/LDC2015E29_DEFT_Rich_ERE_English_Training_Annotation_V2/data -o data/ERE_converted -b bert-large-cased -c /shared/.cache/transformers -l english
```

Arguments:

- -i, --input: Path to the input directory. Should be `data/LDC2015E29/data`.
- -o, --output: Path to the output directory. Should be `data/ERE_converted`.
- -b, --bert: Bert model name.
- -c, --cache\_dir: Path to the BERT cache directory.
- -l, --lang: Language (options: english, spanish).

### Getting the SRL output

After preprocessing, we run the SRL model developed by [Celine Lee](https://github.com/celine-lee). The SRL output will be used by the subsequent event extracion components.

To run the SRL models, you can clone the SRL repos below and follow their instructions:

- [Verb SRL](https://github.com/celine-lee/transformers-srl)
- [Nominal SRL](https://github.com/celine-lee/nominal-srl-allennlpv0.9.0)

You should put the SRL output files under `data/SRL_output/ACE` or `data/SRL_output/ERE`. The output files should be named `{nom|verb}SRL_{split}.jsonl`. For example, `data/SRL_output/ACE/nomSRL_dev.jsonl`, or `data/SRL_output/ERE/nomSRL_all.jsonl`.

To see what format the output files should be, please refer to the sample files: `data/SRL_output/ACE/{nom|verb}SRL_dev_sample.jsonl`. Each sample file has the NomSRL/VerbSRL output respectively. All json fields except for 
`verbs/nominals` come from the preprocessed data in the previous step. The same format applies to other splits and the ERE corpus.

Each sample file has only one sentence, due to ACE confidentiality restrictions.

### Set the configuration
Go to `source/config`, and set the configuration in `config.json`. See `source/config/config_README.md` for details on each parameter.


### Make predictions & Evaluate the model
Run `python source/predict_evaluate.py`. The gold and predicted events, as well as the scores, will be printed to std output. A json version of the predicted events will also be saved to `output_dir/`, in the same format as the input file.


## Citation
If you find this repo useful, please cite the following paper:

```
@inproceedings{lyu-etal-2021-zero,
    title = "Zero-shot Event Extraction via Transfer Learning: {C}hallenges and Insights",
    author = "Lyu, Qing  and
      Zhang, Hongming  and
      Sulem, Elior  and
      Roth, Dan",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-short.42",
    doi = "10.18653/v1/2021.acl-short.42",
    pages = "322--332",
    abstract = "Event extraction has long been a challenging task, addressed mostly with supervised methods that require expensive annotation and are not extensible to new event ontologies. In this work, we explore the possibility of zero-shot event extraction by formulating it as a set of Textual Entailment (TE) and/or Question Answering (QA) queries (e.g. {``}A city was attacked{''} entails {``}There is an attack{''}), exploiting pretrained TE/QA models for direct transfer. On ACE-2005 and ERE, our system achieves acceptable results, yet there is still a large gap from supervised approaches, showing that current QA and TE technologies fail in transferring to a different domain. To investigate the reasons behind the gap, we analyze the remaining key challenges, their respective impact, and possible improvement directions.",
}
```


<!-- LICENSE -->
## License

Distributed under the MIT License.