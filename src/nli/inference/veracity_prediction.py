"""Veracity prediction for claimdecomp.

python3 veracity_prediction.py --test_path path to test file
-- model_path path to model --questions_path path to decomposed questions from claim
-- output_path output/...
"""
import json
import argparse
import random
import torch
import tqdm

from typing import Dict, List
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
from torch import Tensor
from QuanTemp.code.utils.data_loader import read_json
from sentence_transformers import SentenceTransformer

from QuanTemp.code.utils.load_veracity_predictor import VeracityClassifier
from QuanTemp.code.utils.similarity import SimilarityFetch
reranker = SimilarityFetch()

##Set device
if torch.cuda.is_available():
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")

import tqdm

# random.seed(77)
dir_path = os.path.dirname(os.path.realpath(os.getcwd()))


def get_verification(config):
    """Get veracity predictions."""
    key = 'snippet'

    corpus = read_json(config["corpus_path"])
    facts = read_json(config["test_path"])

    qrels = read_json(config["qrels_path"])
    output_path = config["output_path"]
    output_dir = "/".join(output_path.split("/")[:-1])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created successfully.")

    # decomposed_questions = pd.read_csv(
    #     config["questions_path"], sep="@"
    # )

    model_name = config["model_path"]

    nli_model = VeracityClassifier(
        base_model=config["base_model"], model_name=model_name,device=device
    )
    nli_model.model.eval()

    results = []
    matches = 0
    unmatches = 0
    verdicts = {"claim": [],"input":[], "verdict": [],'confidence':[]}


    for index, fact in tqdm.tqdm(enumerate(facts),total=len(facts)):        

        result = {"evidences": []}

        claim = fact["claim"]
        if "speaker" in fact.keys() and fact["speaker"]:
                claim = fact["speaker"] + " " + claim
        if "stated_in" in fact.keys() and fact["stated_in"]:
            claim = claim + " " + fact["stated_in"]

        result["claim"] = claim
        top_k_docs = []
        if(str(index) in qrels):
            top_k_docs = [corpus[idx]['title']+' '+corpus[idx]['snippet'] for idx in list(qrels[str(index)].keys())[:3]]


        verdicts["claim"].append(fact["claim"])
        if len(top_k_docs) > 0:
            for doc in top_k_docs:
                result["evidences"].append(doc)
            input = (
                "[Claim]:"+claim+" [Evidences]: "+"[SEP]".join(top_k_docs)
            )
            if("fact" in input.lower() and "check" in input.lower()):
                print("Here")
            pred_label, confidence = nli_model.predict(input, max_legnth=256)
        elif len(top_k_docs) == 0:
            print("No documents retrieved verifying claim directly")
            pred_label,confidence = nli_model.predict("[Claim]:"+claim+" [Evidences]: ")
            input = ""
        # pred_label = pred_label if abs(probs[1]-probs[0]) > 0.2 else "NONE"
        print("pred_label", pred_label)
        if pred_label == "SUPPORTS":
            verdict = "True"
        elif pred_label == "REFUTES":
            verdict = "False"
        elif pred_label == "CONFLICTING":
            verdict = "Conflicting"

        print("Verdict:", verdict)
        verdicts["verdict"].append(verdict)
        verdicts["input"].append(input)
        verdicts['confidence'].append(confidence[0].tolist())
        results.append(result)
        verdict_1 = pd.DataFrame(verdicts)
        print(verdict_1)
        
        verdict_1.to_csv(f"{output_path}.csv", index=False)
        print(f"{fact['claim']}\t{fact['label']}\t{verdict}")
        if(fact['label']!="False" and fact['label']!="True"):
            actual_label = "Conflicting"
        else:
            actual_label = fact['label']
        if verdict == actual_label:
            matches += 1 
        else:
            unmatches += 1
        print("accuracy", matches / (matches + unmatches))
        with open(f"{output_path}.json", "w") as f:
            json.dump(results, f, indent=4, sort_keys=True)


parser = argparse.ArgumentParser()

parser.add_argument("--test_path", type=str,
                    default="dataset_collection/quantemp_artifacts/final_setup/test.json",
                    help="Path to the test data")
parser.add_argument("--corpus_path", type=str,
                    default="dataset_collection/quantemp_artifacts/final_setup/corpus_1.json", 
                    help="Path to the corpus")
parser.add_argument("--qrels_path", type=str,
                    default=f"dataset_experiments/snippets/retrieval/out/mpnet/test/oracle_combmax.json",
                    help="Path to the corpus")
parser.add_argument("--base_model", type=str,
                    default="FacebookAI/roberta-large-mnli", help="Path to the base model")
parser.add_argument("--model_path", type=str,
                    default="dataset_experiments/snippets/nli/out/train/final_setup_21May_1/qgen_combmax",
                    help="Path to the model")
# parser.add_argument("--model_path", type=str,
#                     default="QuantCheck/model/model_roberta_large_claimdecomp_final/model_weights.zip",
#                     help="Path to the model")
parser.add_argument("--output_path", type=str,
                    default=f"dataset_experiments/snippets/nli/out/inference/misc",
                    help="Path to the output predictions")
args = parser.parse_args()

CONFIG = {
    "corpus_path": args.corpus_path,
    "base_model": args.base_model,
    "model_path": args.model_path,
    "test_path": args.test_path,
    "qrels_path": args.qrels_path,
    "output_path": args.output_path,
}
get_verification(CONFIG)
