import numpy as np
from itertools import product

def postprocess_qa_predictions(input_ids,
                               predictions, # Tuple[np.ndarray, np.ndarray],
							   question_len,
                               version_2_with_negative: bool = True,
                               n_best_size: int = 5,
                               max_answer_length: int = 30,
                               null_score_diff_threshold: float = 0.0,
                               ):
	"""
	Adapted from huggingface utils_qa.py.
	Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
	original contexts. This is the base postprocessing functions for models that only return start and end logits.
	Args:
		predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
			The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
			first dimension must match the number of elements of :obj:`features`.
		version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`True`):
			Whether or not the underlying dataset contains examples with no answers.
		n_best_size (:obj:`int`, `optional`, defaults to 5):
			The total number of n-best predictions to generate when looking for an answer.
		max_answer_length (:obj:`int`, `optional`, defaults to 30):
			The maximum length of an answer that can be generated. This is needed because the start and end predictions
			are not conditioned on one another.
		null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
			The threshold used to select the null answer: if the best answer has a score that is less than the score of
			the null answer minus this threshold, the null answer is selected for this example (note that the score of
			the null answer for an example giving several features is the minimum of the scores for the null answer on
			each feature: all features must be aligned on the fact they `want` to predict a null answer).
			Only useful when :obj:`version_2_with_negative` is :obj:`True`.
	"""

	prelim_predictions = []

	assert len(predictions) == 2, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
	start_logits, end_logits = predictions
	start_logits = [float(_) for _ in start_logits]
	end_logits = [float(_) for _ in end_logits]

	# Update minimum null prediction.
	null_score = start_logits[0] + end_logits[0]
	null_prediction = {
		"span": (0, 0),
		"answer": "",
		"confidence": null_score,
		"start_logit": start_logits[0],
		"end_logit": end_logits[0],
	}

	# Go through all possibilities for the `n_best_size` greater start and end logits.
	start_indices = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
	end_indices = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
	for start_index, end_index in product(start_indices, end_indices):
		# Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
		# to part of the input_ids that are not in the context.

		if start_index >= len(input_ids)-1 or end_index >= len(input_ids)-1 :
			continue

		# Don't add null prediction here.
		if start_index == 0 and end_index == 0:
			continue

		# Don't consider answers with a length that is either < 0 or > max_answer_length.
		if end_index < start_index or end_index - start_index + 1 > max_answer_length:
			continue

		# Answer includes tokens before the context
		if end_index <= question_len or start_index <= question_len:
			continue

		# Answer includes the last special token
		if start_index == len(input_ids) or end_index == len(input_ids):
			continue

		prelim_predictions.append(
			{
				"answer": "non-empty",
				"span": (start_index, end_index),
				"confidence": start_logits[start_index] + end_logits[end_index],
				"start_logit": start_logits[start_index],
				"end_logit": end_logits[end_index],
			}
		)

	if version_2_with_negative:
		# Add the minimum null prediction
		prelim_predictions.append(null_prediction)

	# Only keep the best `n_best_size` predictions.
	all_predictions = sorted(prelim_predictions, key=lambda x: x["confidence"], reverse=True)[:n_best_size]

	# Add back the minimum null prediction if it was removed because of its low score.
	if version_2_with_negative and not any(p["span"] == (0, 0) for p in all_predictions):
		all_predictions.append(null_prediction)

	# Use the offsets to gather the answer text in the original context.
	# for pred in all_predictions:
	# 	span = pred["span"]
	# 	start = span[0]
	# 	end = span[1] + 1
	# 	pred["answer"] = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[start:end]))

	# In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
	# failure.
	if len(all_predictions) == 0 or (len(all_predictions) == 1 and all_predictions[0]["answer"] == ""):
		all_predictions.insert(0, {"answer": "empty", "span": (0, 0), "start_logit": 0.0, "end_logit": 0.0, "confidence": 0.0})

	# Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
	# the LogSumExp trick).
	scores = np.array([pred.pop("confidence") for pred in all_predictions])
	exp_scores = np.exp(scores - np.max(scores))
	probs = exp_scores / exp_scores.sum()

	# Include the probabilities in our predictions.
	for prob, pred in zip(probs, all_predictions):
		pred["confidence"] = prob

	return all_predictions, null_prediction
