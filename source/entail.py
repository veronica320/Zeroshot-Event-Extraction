from transformers import AutoTokenizer, AutoModelForSequenceClassification
import ipdb


te_model_name = 'textattack/xlnet-base-cased-MNLI'
cache_dir = "/shared/.cache/transformers"
te_model = AutoModelForSequenceClassification.from_pretrained(te_model_name, cache_dir=cache_dir).to('cuda:0')
tokenizer = AutoTokenizer.from_pretrained(te_model_name, cache_dir=cache_dir)


def entailment(premise, hypothesis, add_neutral):
	x = tokenizer.encode(premise, hypothesis, return_tensors='pt', max_length=tokenizer.max_len, truncation_strategy='only_first').to('cuda:0')
	logits = te_model(x)[0]
	print(premise, hypothesis)
	print(logits)
	if add_neutral: # take into account the probability of the neutral class when computing softmax
		entail_contradiction_logits = logits[:, [0, 1, 2]]
		probs = entail_contradiction_logits.softmax(1)
		prob_label_is_true = float(probs[:, 1])
	else:
		entail_contradiction_logits = logits[:, [0, 1]]
		probs = entail_contradiction_logits.softmax(1)
		prob_label_is_true = float(probs[:, 1])
	print(probs)
	print(prob_label_is_true)

add_neutral = False
premise = "Syria assassination of Lebanon 's former prime minister"
hypothesis = "Someone is accused of a crime."
entailment(premise, hypothesis, add_neutral)

hypothesis = "Someone is killed."
entailment(premise, hypothesis, add_neutral)

# add_neutral = True
# hypothesis = "Someone is killed."
# entailment(premise, hypothesis, add_neutral)
#
# hypothesis = "Someone is executed."
# entailment(premise, hypothesis, add_neutral)