import json
import uuid
import random
import tqdm

from Doc2Query import Doc2Query


#Change case here to generate for each split
with open("data/test.json") as fp:
    dataset = json.load(fp)


        

query = Doc2Query(api_key="xx",system_prompt_path='src/data_pipe/agq/open-ai/prompt_template.txt')
samples = []
failure_ids  = []
for datapoint in tqdm.tqdm(dataset,total=len(dataset)):
    queries = []
    claim = datapoint["claim"]
    unique_id = uuid.uuid1()
    datapoint["id"] = str(unique_id)
    if "speaker" in datapoint.keys() and datapoint["speaker"]:
        claim = datapoint["speaker"] + " " + claim
    if "stated_in" in datapoint.keys() and datapoint["stated_in"]:
        claim = claim + " " + datapoint["stated_in"]
    passage = datapoint["doc"]
    try:
        queries = query.generate_queries(passage, claim)        
        datapoint["doc_queries"] = queries
    except Exception as e:
        failure_ids.append(datapoint['id'])    
    samples.append(datapoint)
    with open('src/data_pipe/agq/open-ai/out/test/test_agq_backup.jsonl', 'a') as f:
        f.write(json.dumps(datapoint))
        f.write('\n')

print("failures:")
for id in failure_ids:
    print(id)

with open("src/data_pipe/agq/open-ai/out/test/test_agq.json",'w+') as fp:
    json.dump(samples,fp)




# //alpha with BERT score