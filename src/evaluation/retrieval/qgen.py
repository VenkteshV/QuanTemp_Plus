import json
import sys
import pandas as pd


from src.evaluation.retrieval.Retriever import BiEncoderContriever
import evaluator
from src.data_pipe.relevance.relevance import MMR, RelevanceFilter

scorer_mmr = MMR("paraphrase-MiniLM-L6-v2",beta=0.8)
diversity_filter = RelevanceFilter(scorer=scorer_mmr,threshold=0.5,alpha=0.8)

def normalize_dict_values(data):
    if(len(data)<=1):
        return data
    # Find the maximum and minimum values in the dictionary
    max_val = max(data.values())
    min_val = min(data.values())
    
    # Normalize each value in the dictionary
    normalized_data = {key: (value - min_val) / (max_val - min_val) for key, value in data.items()}
    
    return normalized_data

split = 'test'
            
with open('data/corpus.json') as fp:
    corpus = json.load(fp)

with open(f'data/{split}_qrels.json') as fp:
    qrels = json.load(fp)

with open(f'data/{split}.json') as fp:
    claims = json.load(fp)

qgen = pd.read_csv('src/qgen/infer/out/test.csv')
generated_queries = [op.split('[SEP]') for op in qgen['gpt_out'].values]
generated_queries_filtered = []

for i,gen_q in enumerate(generated_queries):
    datapoint = claims[i]
    claim = datapoint["claim"] 
        
    if "speaker" in datapoint.keys() and datapoint["speaker"]:
        claim = datapoint["speaker"] + " " + claim
    if "stated_in" in datapoint.keys() and datapoint["stated_in"]:
        claim = claim + " " + datapoint["stated_in"]
    diverse_queries = diversity_filter.filter(claim,gen_q)
    generated_queries_filtered.append(gen_q)



retriever = BiEncoderContriever(corpus_model_name='facebook/contriever')


corpus_texts = [corpus[idx]['title'] + ' ' +corpus[idx]['snippet'] for idx in corpus.keys()]
retriever.index_corpus(corpus_texts,'data/corpus.index')
# retriever.save_index()

final_results = {}
empty=0
for i in range(len(claims)):
    qqrel = {}
    queries = generated_queries_filtered[i]
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
        for doc_id,score in sorted_documents:
                qqrel[str(doc_id)] = score

    if(not(qqrel)):
        empty+=1

    final_results[str(i)] = qqrel


final_results = {k:v for k,v in final_results.items() if v} 
qrels =  {k:v for k,v in qrels.items() if v} 


res = evaluator.evaluate(qrels,final_results,k_values=[1,5,10,100])
mrr_score = evaluator.calculate_mrr(final_results,qrels)
print(res)
print(mrr_score)





