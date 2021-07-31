"""Scorer of model predictions. Adaped from OneIE. """


def safe_div(num, denom):
	if denom > 0:
		if num / denom <= 1:
			return num / denom
		else:
			return 1
	else:
		return 0


def compute_f1(predicted, gold, matched):
	precision = safe_div(matched, predicted)
	recall = safe_div(matched, gold)
	f1 = safe_div(2 * precision * recall, precision + recall)
	return precision, recall, f1


def convert_arguments(triggers, roles):
	args = set()
	for role in roles:
		trigger_idx = role[0]
		trigger_label = triggers[trigger_idx][-1]
		args.add((trigger_label, role[1], role[2], role[3]))
	return args


def score_graphs(gold_graphs, pred_graphs):
	gold_arg_num = pred_arg_num = arg_idn_num = arg_class_num = 0
	gold_trigger_num = pred_trigger_num = trigger_idn_num = trigger_class_num = 0
	gold_men_num = pred_men_num = men_match_num = 0

	for gold_graph, pred_graph in zip(gold_graphs, pred_graphs):

		# Trigger
		gold_triggers = gold_graph.triggers
		pred_triggers = pred_graph.triggers
		gold_trigger_num += len(gold_triggers)
		pred_trigger_num += len(pred_triggers)
		for trg_start, trg_end, event_type in pred_triggers:
			matched = [item for item in gold_triggers
			           if item[0] == trg_start and item[1] == trg_end]
			if matched:
				trigger_idn_num += 1
				if matched[0][-1] == event_type:
					trigger_class_num += 1

		# Argument
		gold_args = convert_arguments(gold_triggers, gold_graph.roles)
		pred_args = convert_arguments(pred_triggers, pred_graph.roles)
		gold_arg_num += len(gold_args)
		pred_arg_num += len(pred_args)
		for pred_arg in pred_args:
			event_type, arg_start, arg_end, role = pred_arg
			gold_idn = {item for item in gold_args
			            if item[1] == arg_start and item[2] == arg_end
			            and item[0] == event_type}
			if gold_idn:
				arg_idn_num += 1
				gold_class = {item for item in gold_idn if item[-1] == role}
				if gold_class:
					arg_class_num += 1

	trigger_id_prec, trigger_id_rec, trigger_id_f = compute_f1(
		pred_trigger_num, gold_trigger_num, trigger_idn_num)
	trigger_prec, trigger_rec, trigger_f = compute_f1(
		pred_trigger_num, gold_trigger_num, trigger_class_num)
	role_id_prec, role_id_rec, role_id_f = compute_f1(
		pred_arg_num, gold_arg_num, arg_idn_num)
	role_prec, role_rec, role_f = compute_f1(
		pred_arg_num, gold_arg_num, arg_class_num)

	print('Trigger Identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
		trigger_id_prec * 100.0, trigger_id_rec * 100.0, trigger_id_f * 100.0))
	print('Trigger Classification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
		trigger_prec * 100.0, trigger_rec * 100.0, trigger_f * 100.0))
	print('Argument Identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
		role_id_prec * 100.0, role_id_rec * 100.0, role_id_f * 100.0))
	print('Argument Classification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
		role_prec * 100.0, role_rec * 100.0, role_f * 100.0))

	scores = {
		'TC': {'prec': trigger_prec, 'rec': trigger_rec, 'f': trigger_f},
		'TI': {'prec': trigger_id_prec, 'rec': trigger_id_rec,
		       'f': trigger_id_f},
		'AC': {'prec': role_prec, 'rec': role_rec, 'f': role_f},
		'AI': {'prec': role_id_prec, 'rec': role_id_rec, 'f': role_id_f},
	}
	return scores