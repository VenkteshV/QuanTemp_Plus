import json

import tqdm

from src.data_pipe.relevance.relevance import FactCheckFilter, HfScorer, RelevanceFilter
scorer = HfScorer('sentence-transformers/all-mpnet-base-v2')
relevance_filter = RelevanceFilter(scorer=scorer,threshold=0.45,alpha=None)
filter = FactCheckFilter("data/fact_checkers.json")

distractor_corpus = {}
count = 0

with open("src/data_pipe/serp/out/test_serp.json") as fp:
    dataset = json.load(fp)

for data in tqdm.tqdm(dataset,total = len(dataset)):
    search_results = data["google_search"] 
    claim = data['claim']
    claim_url = data['url']
    data["filtered_search_results"]  = {}
    for query,results in search_results.items():
        claim_rel = query+'[SEP]'+claim
        result_snippets = []
        for result in results:
            if('link' not in result.keys() or filter.is_not_fact_checker(claim_url,result["link"])):
                snip = result['title'] if 'title' in result.keys() else ''
                snip+='[SEP]'
                snip += result['snippet'] if 'snippet' in result.keys() else ''
                if('title' in result.keys() or 'snippet' in result.keys()):
                    result_snippets.append(snip)
        filtered_results = relevance_filter.filter(claim_rel,result_snippets)
        filtered_titles = [x.split('[SEP]')[0] for x in filtered_results]
        data["filtered_search_results"][query] = [result for result in results if result['title'] in filtered_titles]
        


with open("src/data_pipe/serp/out/test_serp_filtered.json",'w+') as fp:
    json.dump(dataset,fp,indent=4)





