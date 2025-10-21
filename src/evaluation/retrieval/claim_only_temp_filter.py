import json


from dataset_experiments.snippets.retrieval.final_setup.Retriever import BiEncoderContrieverTemporal
import evaluator


with open('data/corpus.json') as fp:
    corpus = json.load(fp)


retriever = BiEncoderContrieverTemporal(model_name='facebook/contriever')
corpus_ip = [(corpus[idx]['title'] + ' ' +corpus[idx]['snippet'],corpus[idx]['date'])  for idx in corpus.keys()]
retriever.index_corpus(corpus_ip,'data/corpus.index')

for split in ['test']:
    print("Running..",split)


    with open(f'data/{split}_qrels.json') as fp:
        qrels = json.load(fp)

    with open(f'data/{split}.json') as fp:
        claims_all = json.load(fp)


    queries = [(claim['claim'],claim['published']) for claim in claims_all]


    retrieval_results = retriever.retrieve(queries,2048,0.5,verbose=True)


    retrieval_results = {k:v for k,v in retrieval_results.items() if v}
    qrels = {k:v for k,v in qrels.items() if v}


    res = evaluator.evaluate(qrels,retrieval_results,k_values=[1,5,10,100])
    mrr_score = evaluator.calculate_mrr(retrieval_results,qrels)
    print(res)
    print(mrr_score)
