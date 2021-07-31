## Utils for loading the probe lexicon, event ontology, etc. ##

def load_trg_probe_lexicon(fr):
	"""Loads the trigger probe lexicon.
	"""
	lexicon = {}
	for line in fr:
		line = line.strip()
		if line:
			if line.isupper():
				event_type = line
			else:
				lexicon[event_type] = line

	return lexicon

def load_arg_map(fr):
	"""Loads the mapping from SRL arg names to ACE arg names.
	"""
	arg_map = {}
	for line in fr:
		line = line.strip()
		if line:
			if line.isupper():
				event_type = line
				arg_map[event_type] = {}
			else:
				srl_arg, ace_args = line.split(':')[0],line.split(':')[1]
				ace_args = [arg for arg in ace_args.split(',') if '+' not in arg]
				arg_map[event_type][srl_arg] = ace_args
	return arg_map

def load_arg_probe_lexicon(fr, arg_probe_type):
	"""Loads the argument probe lexicon.
	"""
	probe_lexicon = {}
	for line in fr:
		line = line.strip()
		if line:
			if line.isupper():
				event_type = line
				probe_lexicon[event_type] = {}
			else:
				arg, probe = line.split(':')[0], line.split(':')[1]
				if arg_probe_type == 'auto_issth':
					probe = probe + ' is {}.'
				if arg_probe_type == 'auto_sthis':
					probe = '{} is ' + probe[0].lower() + probe[1:].lower() + '.'
				probe_lexicon[event_type][arg] = probe
	return probe_lexicon

