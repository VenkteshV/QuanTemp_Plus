import json

from src.evaluation.retrieval.Retriever import BiEncoderContrieverTemporal
import tqdm
import time
import memory_profiler
import psutil
import GPUtil

import evaluator

print("Starting")
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

corpus_texts = [(corpus[idx]['title'] + ' ' +corpus[idx]['snippet'],corpus[idx]['date'])  for idx in corpus.keys()]



case_map = {'pgmfc_questions':'pgmfc','decomp_questions':'claimdecomp','in_file':'oracle'}
retriever = BiEncoderContrieverTemporal(model_name='facebook/contriever')


retriever.index_corpus(corpus_texts,'data/corpus.index')


for split in ['test']:
    for aggr_case in case_map.keys(): 
        print("Running..",aggr_case,split)

        with open(f'data/{split}_qrels.json') as fp:
            qrels = json.load(fp)

        with open(f'data/{split}.json') as fp:
            claims_train = json.load(fp)

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
                    # scores = retrieval_results[result]
                    scores = normalize_dict_values(retrieval_results[result])
                    for doc_id in scores:
                        if(doc_id in doc_scoring):
                            doc_scoring[doc_id] = max(scores[doc_id],doc_scoring[doc_id])
                        else:
                            doc_scoring[doc_id] = scores[doc_id]

                sorted_documents = sorted(doc_scoring.items(), key=lambda x: x[1], reverse=True)
                for doc_id,score in sorted_documents:
                        qqrel[str(doc_id)] = score

            if(not(qqrel)):
                empty+=1

            final_results[str(i)] = qqrel


        final_results = {k:v for k,v in final_results.items() if v} 
        qrels =  {k:v for k,v in qrels.items() if v} 




        res = evaluator.evaluate(qrels,final_results,k_values=[1,3,5,10,100])
        mrr_score = evaluator.calculate_mrr(final_results,qrels)
        print(res)
        print(mrr_score)
        



