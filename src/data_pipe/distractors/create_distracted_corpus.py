import json

import tqdm

from src.data_pipe.relevance.relevance import FactCheckFilter

with open('src/data_pipe/distractors/out/val_enriched_distractors.json') as fp:
     val = json.load(fp)

with open('src/data_pipe/distractors/out/retrieval_snippets_2/train_enriched_distractors.json') as fp:
     train = json.load(fp)

with open('src/data_pipe/distractors/out/retrieval_snippets_2/test_enriched_distractors.json') as fp:
     test = json.load(fp)

with open('src/data_pipe/distractors/out/retrieval_snippets_2/corpus.json') as fp:
     corpus = json.load(fp)

factcheck_filter = FactCheckFilter("data/fact_checkers.json")
corpus_count = len(corpus)

distractor_count = 0
overlap = 0
corpus_map = {evidence['snippet']:idx for idx,evidence in corpus.items()}

def add_distractors_to_corpus(dataset,corpus):
    global corpus_count
    global distractor_count
    global overlap
    global corpus_map
    for data in tqdm.tqdm(dataset,total=len(dataset)):
        for result in data['claim_google_search']:
            if('link' not in result.keys() or factcheck_filter.is_not_fact_checker(data['url'],result['link'])):
                if('snippet' in result and result['snippet'] in corpus_map):
                    overlap+=1
                    continue                    
                evidence = {}
                evidence['snippet'] = result['snippet'] if 'snippet' in result else ''
                evidence['title'] = result['title'] if 'title' in result else ''
                evidence['link'] = result['link'] if 'link' in result else ''
                evidence['date'] = result['date'] if 'date' in result else ''
                if('snippet' in result):
                    corpus_map[result['snippet']] = corpus_count
                if(str(corpus_count) in corpus):
                     raise Exception("Not supposed to happen")
                corpus[str(corpus_count)] = evidence
                corpus_count+=1
                distractor_count+=1
    return corpus


corpus = add_distractors_to_corpus(train,corpus)
corpus = add_distractors_to_corpus(test,corpus)
corpus = add_distractors_to_corpus(val,corpus)


with open('data/corpus.json','w+') as fp:
     json.dump(corpus,fp)

    

