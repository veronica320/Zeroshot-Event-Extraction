# import ujson as json
import json
import random
from tqdm import tqdm


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
    print('Number of docs:', len(sentence_by_doc))
    for tmp_doc in tqdm(sentence_by_doc):
        # print(tmp_doc)
        # print('Number of sentences:', len(sentence_by_doc[tmp_doc]))
        for tmp_example in sentence_by_doc[tmp_doc]:
            old_sentence = tmp_example['sentence']
            old_tokens = tmp_example['tokens']
            local_event_types = list()
            for tmp_e in tmp_example['event_mentions']:
                local_event_types.append(tmp_e['event_type'])
            local_event_types = set(local_event_types)
            local_sentences = list()
            all_tokens = list()
            # print(local_event_types)
            for tmp_example_2 in sentence_by_doc[tmp_doc]:
                same_event_types = False
                for tmp_event_2 in tmp_example_2['event_mentions']:
                    if tmp_event_2['event_type'] in local_event_types:
                        same_event_types = True
                        break
                if same_event_types:
                    continue
                local_sentences.append(tmp_example_2['sentence'])
                # print(all_sentences)
                all_tokens += tmp_example_2['tokens']
            local_sentences.append(old_sentence)
            # all_tokens.append(old_tokens)
            new_example = dict()
            for tmp_key in tmp_example:
                new_example[tmp_key] = tmp_example[tmp_key]

            new_example['original_sentence'] = old_sentence
            new_example['original_tokens'] = old_tokens
            new_example['sentence'] = ' '.join(local_sentences)
            new_example['tokens'] = all_tokens
            final_result.append(new_example)
    random.shuffle(final_result)
    with open(folder_name + '/modified.' + file_name, 'w') as f:
        for tmp_example in final_result:
            f.write(json.dumps(tmp_example))
            f.write('\n')

generate_new_data('dev.event.json')
generate_new_data('train.event.json')
generate_new_data('test.event.json')

print('end')
