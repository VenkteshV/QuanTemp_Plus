import json

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
from sentence_transformers import SentenceTransformer
import openai

from QuanTemp.code.utils.data_loader import read_json


model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
dir_path = os.path.dirname(os.path.realpath(os.getcwd()))


class QuantVerify:
    def __init__(self):
        #openai.api_key = get_env_var("OPENAI_KEY")
        openai.api_key = "key"
        openai.api_type = 'open_ai'
        openai.api_base = 'https://api.openai.com/v1'
        openai.api_version = None


    def quantverify(self, system_prompt,user_prompt):



        try:
            response = openai.ChatCompletion.create(
            engine="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt,
                            }],
                    temperature=0.3,
                    max_tokens=268,
                    top_p=1.0,
                    frequency_penalty=0.8,
                    presence_penalty=0.6
                )
            return response['choices'][0]['message']['content']
        except Exception as e:
            print("exception",e)

            openai.api_key = "key"
            openai.api_type = 'open_ai'
            openai.api_base = 'https://api.openai.com/v1'
            openai.api_version = None
            response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt,
                            }],
                    temperature=0.3,
                    max_tokens=268,
                    top_p=1.0,
                    frequency_penalty=0.8,
                    presence_penalty=0.6
                )
            output = response['choices'][0]['message']['content']
        return output


def get_top_n_neighbours(sentence, data_emb, data, k, threshold):
    data = list(data)
    sent_emb = model.encode(sentence)
    text_sims = cosine_similarity(data_emb, [sent_emb]).tolist()
    results_sims = zip(range(len(text_sims)), text_sims)
    sorted_similarities = sorted(
        results_sims, key=lambda x: x[1], reverse=True)
    
    top_questions = {}
    for idx, item in sorted_similarities:
        if item[0] > threshold and data[idx]['label'] not in top_questions:
            top_questions[data[idx]['label']] = data[idx]
            if(len(top_questions)==k):
                break
    return top_questions

def map_label(ip_label):
    if ip_label.lower() =="true":
        label = "SUPPORTS"
    elif ip_label.lower() =="false":
        label = "REFUTES"
    else:
        label = "CONFLICTING"
    return label

def create_input(sample,evidences,label=None):
    claim = sample['claim']
    if "speaker" in sample.keys() and sample["speaker"]:
        claim = sample["speaker"] + " " + claim
    if "stated_in" in sample.keys() and sample["stated_in"]:
        claim = claim + " " + sample["stated_in"]

    input = f'[Claim]: {claim}\n'
    input+='[Evidences]: '
    for evidence in evidences:
        input+= f'{evidence}[SEP]\n'
    if(label):
        input+=f'\nLabel: {label}'
    return input


def get_verification():
    facts = read_json(
        "data/test.json"
    )

    few_shot_samples = read_json(
        "data/train.json"
    )

    corpus = read_json(
        "data/corpus.json"
    )

    few_shot_claims = [fact["claim"] for fact in few_shot_samples]
    

    nli_model = QuantVerify()
    results = []
    matches = 0
    unmatches = 0
    verdicts = {"claim": [], "verdict": [],"justification":[]}
    claim_embeddings = model.encode(few_shot_claims)
    

    for index, fact in enumerate(facts[1010:]):

        icl_samples = get_top_n_neighbours(
            fact["claim"], claim_embeddings,few_shot_samples,3,0.3
        )
        prompt = ""
        for label,sample in icl_samples.items():
            evidences_prompt = []


            for query,doc_id in sample['in_file'].items():
                evidences_prompt.append(corpus[str(doc_id)]['title'] + ' ' + corpus[str(doc_id)]['snippet'])

            label = map_label(sample['label'])

            prompt+= '\n'+create_input(sample,evidences_prompt,label)
            

        top_k_docs = [corpus[str(doc_id)]['title'] + ' ' + corpus[str(doc_id)]['snippet'] for doc_id in fact['in_file'].values()]
        
        input = create_input(fact,top_k_docs)
                
        system_prompt = """For the given claim given as [Claim] and given evidences seperated by [SEP] under [Evidences] ,use information from them to fact check the claim and also additionally paying attention fact check by thinking step by step and output the label in the end by performing entailment to fact check claim using the evidence.
        Note: Predict the Label: as strictly one of the following categories: SUPPORTS, REFUTES or CONFLICTING. Label 'SUPPORTS' means the claim is supported by evidences, 'REFUTED' means the claim is refuted by evidences and 'CONFLICTING' means parts of claim is neither fully supported or refuted by the evidences.
        Note: The snippets may be noisy hence use small details, both implicit and explicit to verify the claim.
        Note: You need to verify the claim mentioned by the speaker if present.
        A few examples for this task are as follows: \n"""+prompt
        user_prompt = "Input:\n""" + input+ "For given Input: Give Justification: , Label: "
        print("\nuser_prompt", user_prompt)
        pred_label = nli_model.quantverify(
                system_prompt,user_prompt)
        print("\npred_label***************",pred_label)
        label = pred_label
        justification = pred_label
        if len(pred_label.split("Label")) >1:
            label = pred_label.split("Label")[1]
            justification = pred_label.split("Lebl")[0]
            

        if "SUPPORT" in pred_label or "SUPPORTED" in pred_label:
            verdict = "True"
        elif "REFUTE" in pred_label or "not supported" in pred_label:
            verdict = "False"
        elif "CONFLICTING" in pred_label:
            verdict = "Half True/False"

        verdicts["verdict"].append(verdict)
        verdicts["claim"].append(fact['claim'])
        verdicts["justification"].append(justification)
        verdict_1 = pd.DataFrame(verdicts)
        print(verdict_1)
        verdict_1.to_csv(
            "dataset_experiments/snippets/nli/out/inference/quantempext/gpt3/few_shot_1010_onward.csv", index=False)
        print(f"{fact['claim']}\t{fact['label']}\t{verdict}")
        if verdict == fact["label"]:
            matches += 1
        else:
            unmatches += 1
        print("accuracy", matches/(matches+unmatches))



if __name__ == "__main__":
    get_verification()
