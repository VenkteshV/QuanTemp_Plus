import json


from dataset_experiments.snippets.retrieval.misc.BiEncoderRetriever import BiEncoderContriever
import evaluator

with open('data//corpus.json') as fp:
    corpus = json.load(fp)

with open('data//test_qrels.json') as fp:
    qrels = json.load(fp)

with open('data//test.json') as fp:
    claims_all = json.load(fp)


claims_all_comparison = [claim for claim in claims_all if claim['taxonomy_label']=='comparison']

retriever = BiEncoderContriever(corpus_model_name='facebook/contriever')

corpus_texts = [corpus[idx]['title'] + ' ' +corpus[idx]['snippet'] for idx in corpus.keys()]
retriever.index_corpus(corpus_texts,'dataset_experiments/snippets/retrieval/out/final_setup/corpus.index')

queries = [claim['claim'] for claim in claims_all]


retrieval_results = retriever.retrieve(queries,2048,0.5)

retrieval_results = {k:v for k,v in retrieval_results.items() if v}
qrels = {k:v for k,v in qrels.items() if v}


res = evaluator.evaluate(qrels,retrieval_results,k_values=[1,3,5,10,100])
mrr_score = evaluator.calculate_mrr(retrieval_results,qrels)
print(res)
print(mrr_score)
