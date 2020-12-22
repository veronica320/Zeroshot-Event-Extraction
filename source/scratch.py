from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
import os
import torch

os.chdir('/shared/lyuqing/probing_for_event/')

model_abbr_dict = {"bert":"bert",
                   "bertl":"bert-l",
                   "roberta":"roberta",
                   "robertal":"roberta-l",
                   "xlmr":"xlm-roberta",
                   "xlmrl": "xlm-roberta-l",
                   "mbert":"mbert",
                   "mbert-c":"mbert-cased"
                   }


gpu_devices = [0]
model_dir = 'output_model_dir'
EX_QA_model_type = 'roberta'
EX_idk = False
EX_model_name = model_abbr_dict[EX_QA_model_type]
ex_qa_model_path = f"{model_dir}/qamr{'_idk' if EX_idk else ''}_{EX_model_name}"  # Extractive QA model
ex_qa_model = AutoModelForQuestionAnswering.from_pretrained(ex_qa_model_path).to('cuda:' + str(gpu_devices[0]))
ex_tokenizer = AutoTokenizer.from_pretrained(ex_qa_model_path)

def answer_ex(self, question, context):
	"""Answers an extractive question. Outputs the answer span."""
	if context and len(context) > 1:  # Capitalize the first letter of the premise
		context = context[0].upper() + context[1:]
	question = question[0].upper() + question[1:]
	if question[-1] != '?':
		question = question + '?'

	# input_tensor = self.yn_tokenizer.encode_plus(question, context, return_tensors="pt").to(
	# 	'cuda:' + str(self.gpu_devices[0]))
	# classification_logits = self.yn_qa_model(**input_tensor)[0]
	# probs = torch.softmax(classification_logits, dim=1).tolist()
	# if self.YN_idk:  # class0:Yes, class1:No, class2:IDK
	# 	yes_prob = probs[0][0]
	# else:  # class0:No, class1:Yes
	# 	yes_prob = probs[0][1]

	input_tensor = ex_tokenizer(question, context, add_special_tokens=True, return_tensors="pt").to(
		'cuda:' + str(gpu_devices[0]))
	input_ids = input_tensor["input_ids"].tolist()[0]
	text_tokens = ex_tokenizer.convert_ids_to_tokens(input_ids)
	outputs = ex_qa_model(**input_tensor)
	answer_start_scores = outputs.start_logits
	answer_end_scores = outputs.end_logits
	answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
	answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
	answer = ex_tokenizer.convert_tokens_to_string(ex_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
	print(f"Question: {question}")
	print(f"Answer: {answer}")