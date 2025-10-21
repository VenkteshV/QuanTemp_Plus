import json
from src.evaluation.retrieval.Retriever import BiEncoderRetrieverTemporal
import evaluator

import tqdm
import os
import time
import memory_profiler
import psutil
import GPUtil

print("Starting...")

def normalize_dict_values(data):
    if(len(data)<=1):
        return data
    # Find the maximum and minimum values in the dictionary
    max_val = max(data.values())
    min_val = min(data.values())
    
    # Normalize each value in the dictionary
    normalized_data = {key: (value - min_val) / (max_val - min_val) for key, value in data.items()}
    
    return normalized_data

def print_system_usage():
    # Get RAM usage
    ram_usage = psutil.virtual_memory()
    print(f"RAM Usage: {ram_usage.percent}%")

    # Get GPU usage
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.load*100}% | Memory Used: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")

with open('data/corpus.json') as fp:
    corpus = json.load(fp)


case_map = {'in_file':'oracle'}
# 'tasb':'sentence-transformers/msmarco-distilbert-base-tas-b','ance':'sentence-transformers/msmarco-roberta-base-ance-firstp',
model_map = {'mpnet':'sentence-transformers/all-mpnet-base-v2'}


corpus_texts = [(corpus[idx]['title'] + ' ' +corpus[idx]['snippet'],corpus[idx]['date'])  for idx in corpus.keys()]

split = 'test'
aggr_case = 'in_file'
for model in model_map:
    retriever = BiEncoderRetrieverTemporal(corpus_model_name='sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base',
                                           query_model_name='sentence-transformers/facebook-dpr-question_encoder-single-nq-base')
    retriever.index_corpus(corpus_texts,None)

    with open(f'data/{split}_qrels.json') as fp:
        qrels = json.load(fp)

    with open(f'data/{split}.json') as fp:
        claims_train = json.load(fp)
    # retriever.save_index()
    start_time = time.time()
    mem_usage = memory_profiler.memory_usage(-1)
    start_mem = max(mem_usage)    
    print_system_usage()
    print("Running..",aggr_case,split)

    final_results = {}
    empty=0
    for i in tqdm.tqdm(range(len(claims_train)),total=len(claims_train)):
        qqrel = {}
        queries = [(val,claims_train[i]['published']) for val in claims_train[i][aggr_case]]
        if(not(queries)):
            qqrel ={}
        else:
            retrieval_results = retriever.retrieve(queries,2048,0.5)
            doc_scoring = {}
            for result in retrieval_results:
                scores = normalize_dict_values(retrieval_results[result])
                for doc_id in scores:
                    if(doc_id in doc_scoring):
                        doc_scoring[doc_id] = max(scores[doc_id],doc_scoring[doc_id])
                    else:
                        doc_scoring[doc_id] = scores[doc_id]

            sorted_documents = sorted(doc_scoring.items(), key=lambda x: x[1], reverse=True)
            for doc_id,score in sorted_documents[:100]:
                    qqrel[str(doc_id)] = score

        if(not(qqrel)):
            empty+=1

        final_results[str(i)] = qqrel



    final_results = {k:v for k,v in final_results.items() if v} 
    qrels =  {k:v for k,v in qrels.items() if v} 

    print_system_usage()
    end_time = time.time()
    execution_time = end_time-start_time
    mem_usage = memory_profiler.memory_usage(-1)
    end_mem = max(mem_usage)

    print(model)
    print("Execution time:", execution_time, "seconds")
    print("Memory usage:", end_mem - start_mem, "MB")




    print(aggr_case)
    res = evaluator.evaluate(qrels,final_results,k_values=[1,5,10,100])
    print(res)
    




