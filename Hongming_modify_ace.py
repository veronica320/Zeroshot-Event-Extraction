# import ujson as json
import json
import random


def generate_new_data(file_name):
    folder_name = '/shared/lyuqing/probing_for_event/data/ACE_oneie/en/event_only'

    sentence_by_doc = dict()

    tmp_data = list()
    with open(folder_name+'/'+file_name, 'r') as f:
        for line in f:
            tmp_example = json.loads(line)
            tmp_data.append(tmp_example)

    for tmp_example in tmp_data:
        if tmp_example['doc_id'] not in sentence_by_doc:
            sentence_by_doc[tmp_example['doc_id']] = list()
        sentence_by_doc[tmp_example['doc_id']].append(tmp_example)

    final_result = list()
    for tmp_doc in sentence_by_doc:

        for tmp_example in sentence_by_doc[tmp_doc]:
            old_sentence = tmp_example['sentence']
            old_tokens = tmp_example['tokens']
            event_types = list()
            for tmp_e in tmp_example['event_mentions']:
                event_types.append(tmp_e['event_type'])
            event_types = set(event_types)
            all_sentences = list()
            all_tokens = list()
            for tmp_example_2 in sentence_by_doc[tmp_doc]:
                same_event_types = False
                for tmp_event_2 in tmp_example_2['event_mentions']:
                    if tmp_event_2['event_type'] in event_types:
                        same_event_types = True
                        break
                if same_event_types:
                    continue
                all_sentences.append(tmp_example_2['sentence'])
                all_tokens += tmp_example_2['tokens']
            all_sentences.append(old_sentence)
            all_tokens.append(old_tokens)
            tmp_example['original_sentence'] = old_sentence
            tmp_example['original_tokens'] = old_tokens
            tmp_example['sentence'] = ' '.join(all_sentences)
            tmp_example['tokens'] = all_tokens
            final_result.append(tmp_example)
    random.shuffle(final_result)
    with open(folder_name + '/modified.' + file_name, 'r') as f:
        for tmp_example in final_result:
            f.write(json.dumps(tmp_example))
            f.write('\n')

generate_new_data('train.event.json')
generate_new_data('dev.event.json')
generate_new_data('test.event.json')
