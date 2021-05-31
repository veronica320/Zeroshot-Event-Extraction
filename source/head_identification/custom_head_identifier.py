import os
from allennlp.predictors.predictor import Predictor
from nltk import pos_tag

def identify_head_custom(dependency_parser, span, tokens, pos_tags):
	""" A coarse-grained head identifier. """

	instance = dependency_parser._dataset_reader.text_to_instance(tokens, pos_tags)
	output = dependency_parser.predict_instance(instance)

	start_ix = span[0]

	root_idx = output['predicted_heads'].index(0)
	pos_list = output['pos']
	words_list = output['words']
	parent_idx = 0
	current_idx = root_idx
	siblings = [current_idx]
	pos_in_siblings = 0
	while True:
		if pos_list[current_idx].startswith(('NN', 'PRP', 'CD')):
			word = words_list[current_idx]
			global_idx = start_ix + current_idx
			return global_idx, word
		pos_in_siblings = pos_in_siblings - 1
		if pos_in_siblings >= 0:  # check if there are siblings on the left of the current node
			current_idx = siblings[pos_in_siblings]  # if yes, move to the rightmost sibling
		else:  # if no, move to the rightmost child of the current node
			parent_idx = current_idx + 1
			siblings = [i for i, x in enumerate(output['predicted_heads']) if x == parent_idx]
			if siblings:
				current_idx = siblings[-1]
				pos_in_siblings = len(siblings) - 1
			else:
				return None, None

if __name__ == "__main__":

	texts = ["a person that likes cat",
	         "the old tree from India",
	         "Mary Curie",
	         "U. S.",
	         "MCI to pay largest fine imposed by the SEC on a company",
	         "1.4 billion dollars",
	         "U . S . and British troops"
	         ]

	dependency_parser = Predictor.from_path(
		"https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")

	for text in texts:
		tokens = text.split()
		span = range(len(tokens))
		pos_tags = [tag for _, tag in pos_tag(tokens)]
		head = identify_head_custom(dependency_parser, span, tokens, pos_tags)
		print(f"Text: {text}")
		print(f"POS tags: {pos_tags}")
		print(f"Head: {head}\n")
