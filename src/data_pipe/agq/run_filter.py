import random
import tqdm
import json
import pandas as pd
from sklearn.metrics import accuracy_score

from src.data_pipe.relevance.relevance import MMR, RelevanceFilter


with open("src/data_pipe/agq/open-ai/out/test/test_agq.json") as fp:
    dataset = json.load(fp)

failures = [(idx,record) for idx,record in enumerate(dataset) if "doc_queries" not in record.keys()]
# random.seed(44)
# records = random.sample(dataset,10)

pred = []
scorer = MMR("paraphrase-MiniLM-L6-v2",beta=0.4)
filter = RelevanceFilter(scorer=scorer,threshold=0.5,alpha=0.8)
total_queries= 0
samples = []
for datapoint in tqdm.tqdm(dataset,total=len(dataset)):
    claim = datapoint["claim"]
    if "speaker" in datapoint.keys() and datapoint["speaker"]:
        claim = datapoint["speaker"] + " " + claim
    if "stated_in" in datapoint.keys() and datapoint["stated_in"]:
        claim = claim + " " + datapoint["stated_in"]
    passage = datapoint["doc"]
    queries = datapoint["doc_queries"]
    filtered_queries = filter.filter(claim=claim,doc=passage,queries=queries)
    datapoint["doc_queries_filtered"] = filtered_queries
    relevance_array = []
    for query in queries:
        relevance_array.append(1 if query in filtered_queries else 0)    
    
    datapoint['filter_relevance_array'] = relevance_array
    total_queries+=len(filtered_queries)
    samples.append(datapoint)








# df = pd.DataFrame(samples, columns=["claim", "doc","doc_queries","filter_relevance_array","doc_queries_filtered"])
# attributes_df = pd.DataFrame(df['doc_queries'].tolist(), columns=[f'query_{i+1}' for i in range(len(df['doc_queries'][0]))])
# df = pd.concat([df.drop(['doc_queries'], axis=1), attributes_df], axis=1)
# csv_file_path = "dataset_collection/agq/open-ai/out/output_4/val_sample_filter_7.xlsx"
# df.to_excel(csv_file_path, index=False)

with open("src/data_pipe/agq/open-ai/out/test/test_agq_filtered.json",'w+') as fp:
    json.dump(samples,fp)

print(total_queries)
print(len(samples))

# for query in queries:
#     print(query)



