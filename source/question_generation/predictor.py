import json
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor
from overrides import overrides


@Predictor.register('question_generation')
class QuestionGenerationPredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(context=json_dict['text'],
                                                     start=json_dict['start'],
                                                     end=json_dict['end'])

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        input_dict = outputs['metadata']['input_dict']
        input_dict['question'] = outputs['predicted_question']
        return json.dumps(input_dict) + '\n'