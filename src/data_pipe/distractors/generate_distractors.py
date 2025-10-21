import json
import logging

from QuantCheck.code.utils.similarity import SimilarityFetch
from src.data_pipe.relevance.relevance import FactCheckFilter
from src.data_pipe.serp.SERP import SERPER
import tqdm



quotes = ["\"","\'",'“','”']

def remove_quotes(input_string):
    if(input_string[0]==input_string[-1] and input_string[0] in quotes):
        input_string =  input_string.rstrip('"').rstrip('“').rstrip('”')
        input_string =  input_string.lstrip('"').lstrip('“').lstrip('”')
    return input_string

with open('data/corpus.json') as fp:
    corpus = json.load(fp)


with open('data/val.json') as fp:
    claims_train = json.load(fp)

similarity_model = SimilarityFetch()

my_api_key = 'serpapi_key'
factcheck_filter = FactCheckFilter("data/fact_checkers.json")

# with_quotes  = []
# total_queries = 0

# # for claim in claims_test:
# #      with_quotes.extend(filter(lambda x: remove_quotes(x)!=x,list(claim['in_file'].keys())))
# #      total_queries+=len(claim['in_file'])

# for claim in claims_test:
#      if(remove_quotes(claim['claim'])!=claim['claim']):
#           with_quotes.append(claim['claim'])
        

searcher = SERPER(api_key=my_api_key) 




for sample in tqdm.tqdm(claims_train,total=len(claims_train)):

    claim = remove_quotes(sample['claim'])
    if "speaker" in sample.keys() and sample["speaker"]:
            claim = sample["speaker"] + " " + claim

    filter_date = sample["published"]
    claim_url = sample['url']

    try:
        results = searcher.fetch_results(claim, filter_date)
        sample['claim_google_search'] = results

    except Exception as e:
        print("Unable to fetch results for query", claim,e)

    with open('src/data_pipe/distractors/out/val_enriched_distractors_backup.jsonl', 'a') as f:
            f.write(json.dumps(sample))
            f.write('\n')


# with open('dataset_collection/quantemp_artifacts/retrieval_snippets_2/test_enriched_distractors_backup.jsonl') as fp:
#      for line in fp:
#         samples_serped.append(json.loads(line))

with open('src/data_pipe/distractors/out/val_enriched_distractors.json','w+') as fp:
     json.dump(claims_train,fp)


# count = 0
# with open('dataset_collection/quantemp_artifacts/retrieval_snippets_2/test_enriched_distractors.json') as fp:
#      test = json.load(fp)
    
# corpus_count = len(corpus)

    
# for claim in tqdm.tqdm(test,total=len(test)):

#     for result in claim['claim_google_search']:
#         if('link' not in result or factcheck_filter.is_not_fact_checker(claim['url'],result['link'])):
#             evidence = {}
#             evidence['snippet'] = result['snippet'] if 'snippet' in result else ''
#             evidence['title'] = result['title'] if 'title' in result else ''
#             evidence['link'] = result['link'] if 'link' in result else ''
#             evidence['date'] = result['date'] if 'date' in result else ''
#             corpus[count] = evidence
#             corpus_count+=1

# with open('dataset_collection/quantemp_artifacts/retrieval_snippets_2/corpus_extended_distractors.json','w+') as fp:
#      json.dump(corpus,fp)





    # claim_text = remove_quotes(claim['claim'])
    # if "speaker" in claim.keys() and claim["speaker"]:
    #         claim_text = claim["speaker"] + " " + claim_text
    # result_snippets = [c['title']+' '+(c['snippet'] if 'snippet' in c else '') for c in claim['claim_google_search'] 
    #                    if (('snippet' not in claim['claim'] or claim['claim'] not in c['snippet']) 
    #                        and ('link' not in c or factcheck_filter.is_not_fact_checker(claim['url'],c['link'])))]
    # top_snippets = []
    # if(result_snippets):
    #     snippet_embeddings = similarity_model.model.encode(result_snippets)
    #     top_snippets = similarity_model.get_top_k_similar_instances(claim_text,snippet_embeddings,result_snippets,3,0.5)
    # if(not(top_snippets)):
    #      count+=1
    # claim['claim_only_in_file'] = top_snippets

# with open('dataset_collection/quantemp_artifacts/retrieval_snippets_2/test_enriched_distractors.json','w+') as fp:
#      json.dump(test,fp)




# for sample in samples_serped:
#     filtered_results = []
#     for result in sample['claim_google_search']:
#         if('link' not in result.keys() or factcheck_filter.is_not_fact_checker(sample['url'],result['link'])):
#                   filtered_results.append(result)
#     sample['claim_google_search'] = filtered_results

# with open("dataset_collection/quantemp_artifacts/manual_anotation/samples_filtered.json",'w+') as fp:
#     json.dump(samples_serped,fp)









    



