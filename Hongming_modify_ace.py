# import ujson as json
import json

folder_name = '/shared/lyuqing/probing_for_event/data/ACE_oneie/en/event_only'


tmp_data = list()
with open(folder_name+'/'+'dev.event.json', 'r') as f:
    for line in f:
        tmp_example = json.loads(line)
        tmp_data.append(tmp_example)
    # tmp_data = json.load(f)

for tmp_example in tmp_data:
    print(tmp_example)
    break
