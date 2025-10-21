import json


from src.evaluation.retrieval.Retriever import BiEncoderContrieverTemporal
import evaluator
from collections import Counter
import tqdm
import pandas as pd
from src.data_pipe.relevance.relevance import RelevanceFilter,MMR


scorer_mmr = MMR("paraphrase-MiniLM-L6-v2",beta=0.8)
diversity_filter = RelevanceFilter(scorer=scorer_mmr,threshold=0.5,alpha=0.8)

def normalize_dict_values(data):
    if(len(data)<=1):
        return data
    if(len(set(data.values()))==1):
        return data
    # Find the maximum and minimum values in the dictionary
    max_val = max(data.values())
    min_val = min(data.values())
    
    # Normalize each value in the dictionary
    normalized_data = {key: (value - min_val) / (max_val - min_val) for key, value in data.items()}
    
    return normalized_data

with open('data/corpus.json') as fp:
    corpus = json.load(fp)

splits = ['test']

for split in splits:

    retriever = BiEncoderContrieverTemporal(model_name='facebook/contriever')

    qgen = pd.read_csv(f'src/qgen/infer/out/{split}.csv')
    generated_queries = [op.split('[SEP]') for op in qgen['gpt_out'].values]

    corpus_texts = [(corpus[idx]['title'] + ' ' +corpus[idx]['snippet'],corpus[idx]['date'])  for idx in corpus.keys()]
    retriever.index_corpus(corpus_texts,'data/corpus.index')

    with open(f'data/{split}_qrels.json') as fp:
        qrels = json.load(fp)

    with open(f'data/{split}.json') as fp:
        claims = json.load(fp)


    final_results = {}
    empty=0
    for i in tqdm.tqdm(range(len(claims)),total=len(claims)):
        qqrel = {}
        fact = claims[i]
        claim = fact["claim"]
        if "speaker" in fact.keys() and fact["speaker"]:
                claim = fact["speaker"] + " " + claim
        if "stated_in" in fact.keys() and fact["stated_in"]:
            claim = claim + " " + fact["stated_in"]
        queries_orig = [(val,claims[i]['published']) for val in generated_queries[i]]
        diverse_queries = diversity_filter.filter(claim,[q[0] for q in queries_orig])
        queries = [q for q in queries_orig if q[0] in diverse_queries]
        if(not(queries)):
            qqrel ={}
        else:
            retrieval_results = retriever.retrieve(queries,2048,0.5)
            doc_scoring = {}
            for result in retrieval_results:
                if(retrieval_results[result]):
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

    print("Num empty are",empty)


    final_results = {k:v for k,v in final_results.items() if v} 
    qrels =  {k:v for k,v in qrels.items() if v} 



    res = evaluator.evaluate(qrels,final_results,k_values=[1,5,10,100])
    mrr_score = evaluator.calculate_mrr(final_results,qrels)
    print(res)
    print(mrr_score)



