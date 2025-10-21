import json 
import json
from beir import LoggingHandler
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
import evaluator
from dateutil.parser import parse


import logging

dataset = "quantempext"

with open('data/corpus.json') as fp:
    corpus = json.load(fp)

with open('data/test_qrels.json') as fp:
    qrels = json.load(fp)

with open('data/test.json') as fp:
    claims = json.load(fp)


def normalize_dict_values(data):
    # Find the maximum and minimum values in the dictionary
    max_val = max(data.values())
    min_val = min(data.values())
    
    # Normalize each value in the dictionary
    normalized_data = {key: (value - min_val) / (max_val - min_val) for key, value in data.items()}
    
    return normalized_data




#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

### https://www.elastic.co/
hostname = "https://localhost:9200" #localhost
index_name = "quantempext" # scifact

#### Intialize #### 
# (1) True - Delete existing index and re-index all documents from scratch 
# (2) False - Load existing index
initialize = True # False

#### Sharding ####
# (1) For datasets with small corpus (datasets ~ < 5k docs) => limit shards = 1 
# SciFact is a relatively small dataset! (limit shards to 1)
number_of_shards = 1
model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards,ca_cert="http_ca.crt",basic_auth="pwd")

# (2) For datasets with big corpus ==> keep default configuration
# model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
retriever = EvaluateRetrieval(model,k_values=[1,5,10,100])

#### Retrieve dense results (format of results is identical to qrels)
k=100
corpus_new = {k: {'text': v['snippet'], 'title': v['title'] } for k,v in corpus.items()}
date_map = {}
default_date = parse('01-01-1000')
for k,v in corpus.items():
    try:
        date_map[k] = parse(v['date'])
    except:
        date_map[k] =  default_date

queries = { str(idx): v['claim'] for idx,v in enumerate(claims)}
result_qrels = retriever.retrieve(corpus_new, queries)
    





qrels = {k:v for k,v in qrels.items() if v}
result_qrels = {k:v for k,v in result_qrels.items() if k in qrels.keys()}

#### Evaluate your retrieval using NDCG@k, MAP@K ...
logging.info("Retriever evaluation for k in: {}".format([1,5,10,100]))
results = evaluator.evaluate(qrels, result_qrels, k_values=[1,5,10,100])
print(results)


