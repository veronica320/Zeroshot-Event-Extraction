import os
import stanza
from stanza.server import CoreNLPClient
from nltk import pos_tag

def initialize_head_identifier():
	EN_CUSTOM_PROPS = {'annotators': 'coref',
	                   'tokenize.language': 'en',
	                   'tokenize.white': True,
	                   }
	model_path = "/shared/lyuqing/probing_for_event/env/stanza_resources"
	corenlp_path = "/shared/lyuqing/probing_for_event/env/stanza_corenlp"
	os.environ["STANZA_RESOURCES_DIR"] = model_path
	os.environ["CORENLP_HOME"] = corenlp_path
	stanza_client = CoreNLPClient(timeout=30000,
	                              memory='16G',
	                              properties=EN_CUSTOM_PROPS,
	                              be_quiet=True,
	                             )
	stanza_client.start()
	return stanza_client

def identify_head(stanza_client, span, tokens):

	tokens_wo_whitespace = []
	for token in tokens:
		if ' ' in token:
			token = token.replace(' ','')
		tokens_wo_whitespace.append(token)

	text = ' '.join(tokens_wo_whitespace)
	ann = stanza_client.annotate(text)

	# find the longest mention among all mentions
	longest_mention = None
	longest_span = 0

	for mention in ann.mentionsForCoref:
		if mention.endIndex - mention.startIndex > longest_span:
			longest_span = mention.endIndex - mention.startIndex
			longest_mention = mention

	if longest_mention == None:
		return (None, None)

	sent_id = longest_mention.sentNum
	sent = ann.sentence[sent_id]
	sent_token_offset_begin = sent.tokenOffsetBegin

	start_idx = span[0]
	head_index = longest_mention.headIndex + sent_token_offset_begin
	global_head_index = start_idx + head_index
	head_string = tokens[head_index]
	return (global_head_index, head_string)

def shut_head_identifier(stanza_client):
	stanza_client.stop()

if __name__ == "__main__":

	texts = [
	         "a person that likes cat",
	         "the old tree from India",
	         "Mary Curie",
	         "U. S.",
	         "MCI to pay largest fine imposed by the SEC on a company",
	         "1.4 billion dollars",
	         "U . S . and British troops"
	         ]

	stanza_client = initialize_head_identifier()
	for text in texts:
		tokens = text.split()
		pos_tags = [tag for _, tag in pos_tag(tokens)]
		span = range(len(tokens))
		head = identify_head(stanza_client, span, tokens)

		print(f"Text: {text}")
		print(f"POS tags: {pos_tags}")
		print(f"Head: {head}\n")

	shut_head_identifier(stanza_client)






