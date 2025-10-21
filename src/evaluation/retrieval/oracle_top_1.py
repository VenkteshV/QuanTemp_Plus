import json

from src.evaluation.retrieval.Retriever import BiEncoderContrieverTemporal
import evaluator
from collections import Counter
import tqdm

def normalize_dict_values(data):
    if(len(data)<=1):
        return data
    # Find the maximum and minimum values in the dictionary
    max_val = max(data.values())
    min_val = min(data.values())
    
    # Normalize each value in the dictionary
    normalized_data = {key: (value - min_val) / (max_val - min_val) for key, value in data.items()}
    
    return normalized_data

with open('data/corpus.json') as fp:
    corpus = json.load(fp)


case_map = {'in_file':'oracle'}
retriever = BiEncoderContrieverTemporal(model_name='facebook/contriever')

corpus_texts = [(corpus[idx]['title'] + ' ' +corpus[idx]['snippet'],corpus[idx]['date'])  for idx in corpus.keys()]
retriever.index_corpus(corpus_texts,'data/corpus.index')

split = 'test'
for aggr_case in case_map.keys(): 
        print("Running..",aggr_case,split)

        with open(f'data/{split}_qrels.json') as fp:
            qrels = json.load(fp)

        with open(f'data/{split}.json') as fp:
            claims_train = json.load(fp)
        # retriever.save_index()

        final_results = {}
        empty=0
        for i in tqdm.tqdm(range(len(claims_train)),total=len(claims_train)):
            qqrel = {}
            queries = [(val,claims_train[i]['published']) for val in claims_train[i][aggr_case]]
            if(not(queries)):
                qqrel ={}
            else:
                retrieval_results = retriever.retrieve(queries,2048,0.5)
                for q,res in retrieval_results.items():
                    if(res):
                        top_1 = list(res.keys())[0]
                        qqrel[top_1] = res[top_1]
            if(not(qqrel)):
                empty+=1

            final_results[str(i)] = qqrel


        final_results = {k:v for k,v in final_results.items() if v} 
        qrels =  {k:v for k,v in qrels.items() if v} 

        score = evaluator.calculate_mrr(final_results,qrels)
        print("MRR", score)


        res = evaluator.evaluate(qrels,final_results,k_values=[1,3,5,10,100])
        print(res)




