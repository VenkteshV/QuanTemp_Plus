import json
import logging
import random

from tqdm import tqdm

from SERP import SERPER

my_api_key = 'serp_api_key'


searcher = SERPER(api_key=my_api_key) 

with open("src/data_pipe/agq/open-ai/out/test/test_agq_filtered.json") as fp:
    dataset = json.load(fp)

with open('src/data_pipe/serp/out/final/test_serp_backup.jsonl', 'w+') as f:
    pass


count = 0
samples = []
for datapoint in tqdm(dataset[573:],total=len(dataset)):
    queries = datapoint["doc_queries_filtered"]
    filter_date = datapoint["published"]
    result_dict = {}
    for query in queries:
        try:
            results = searcher.fetch_results(query, filter_date)
            result_dict[query] = results
            count+=1
        except Exception as e:
            logging.info("Unable to fetch results for query", query)
    datapoint["google_search"] = result_dict
    samples.append(datapoint)
    with open('src/data_pipe/serp/out/final/test_serp_backup.jsonl', 'a') as f:
        f.write(json.dumps(datapoint))
        f.write('\n')

print(count,len(dataset))

with open("src/data_pipe/serp/out/final/test_serp.json", "w+") as fp:
    json.dump(samples, fp)





