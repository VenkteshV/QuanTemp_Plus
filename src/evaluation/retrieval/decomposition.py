import json

from src.evaluation.retrieval.Retriever import BiEncoderContriever
import evaluator

def normalize_dict_values(data):
    if(len(data)<=1):
        return data
    # Find the maximum and minimum values in the dictionary
    max_val = max(data.values())
    min_val = min(data.values())
    
    # Normalize each value in the dictionary
    normalized_data = {key: (value - min_val) / (max_val - min_val) for key, value in data.items()}
    
    return normalized_data

# case_map = {'decomp_questions':'claimdecomp','pgmfc_questions':'pgmfc','in_file':'oracle'}
# top_k  = [3,5,7,10]
case = 'decomp_questions'
# for k in top_k:
for split in ['test']:
                
            with open('data/corpus.json') as fp:
                corpus = json.load(fp)

            with open(f'data/{split}_qrels.json') as fp:
                qrels = json.load(fp)

            with open(f'data/{split}.json') as fp:
                claims_train = json.load(fp)

            import sys


            retriever = BiEncoderContriever(corpus_model_name='facebook/contriever')


            corpus_texts = [corpus[idx]['title'] + ' ' +corpus[idx]['snippet'] for idx in corpus.keys()]
            retriever.index_corpus(corpus_texts,'data/contriever_corpus.index')
            # retriever.save_index()

            final_results = {}
            empty=0
            for i in range(len(claims_train)):
                qqrel = {}
                queries = list(claims_train[i][case])
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


            res = evaluator.evaluate(qrels,final_results,k_values=[1,3,5,10,100])
            mrr_score = evaluator.calculate_mrr(final_results,qrels)
            print(res)
            print(mrr_score)
            





