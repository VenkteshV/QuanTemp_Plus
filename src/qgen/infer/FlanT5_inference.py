import transformers
from torch.utils.data import Dataset
import pandas as pd


# import pytorch_lightning as pl

# from pytorch_lightning.callbacks import ModelCheckpoint





import random
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import Dataset, load_metric

import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
import json

nltk.download('punkt')

with open('data//test.json') as fp:
    claims_test = json.load(fp)

with open('data//train.json') as fp:
    claims_train = json.load(fp)

with open('data//val.json') as fp:
    claims_val = json.load(fp)


TOKENIZER_PATH = 'google/flan-t5-large'

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)


raw_datasets = {}
xmetric = load_metric("rouge")



def create_qgen_dataset(dataset):
    qgen_dataset = []
    for fact in dataset:
        record = {}
        claim = fact["claim"]
        if "speaker" in fact.keys() and fact["speaker"]:
                claim = fact["speaker"] + " " + claim
        if "stated_in" in fact.keys() and fact["stated_in"]:
            claim = claim + " " + fact["stated_in"]
        if "published" in fact.keys() and fact["published"]:
            claim = claim + f"Published: {fact['published']}"


        record['claim'] = claim
        record['queries'] = list(fact['in_file'].keys())
        qgen_dataset.append(record)
    return qgen_dataset



train_dataset = Dataset.from_list(create_qgen_dataset(claims_train))
test_dataset = Dataset.from_list(create_qgen_dataset(claims_test))
val_dataset = Dataset.from_list(create_qgen_dataset(claims_val))

model = AutoModelForSeq2SeqLM.from_pretrained("src/qgen/train/out/checkpoint-8400")
model.to('cuda')

"""# Evaluation"""
transformers.set_seed(7)
def generate_output(test_samples, model):
    inputs = tokenizer(
        test_samples,
        max_length=128,
        padding=True,
        return_tensors="pt")

    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    outputs = model.generate(input_ids, attention_mask=attention_mask,min_length = 64, max_length = 128, do_sample=True, top_p=0.95, top_k=50)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return output_str

def genrate_output_batched(test_samples,model,batch_size=16):
    final_op = []
    for i in range(0,len(test_samples),batch_size):  
        print("Progress",i,len(test_samples)) 
        batch_samples = test_samples[i:i+batch_size]
        op = generate_output(batch_samples,model)
        final_op.extend(op)
    return final_op


case_map = {'val':val_dataset,'test':test_dataset,'train':train_dataset}

ip_case_map = {'val':claims_val,'test':claims_test,'train':claims_train}
for case in case_map:
    print(case)
    dataset = case_map[case]
    ip = ip_case_map[case]
    
    
    inputs = ["decompose the compositional claim into queries:"+c for c in dataset['claim']]
    ground = ['[SEP]'.join(q) for q in dataset['queries']]





    seq_list = genrate_output_batched(inputs,model)



    final_test = {"queries":inputs,"gpt_out":seq_list, "ground_truth":ground,"just_doc":[ip[i]['doc'] for i in range(len(ip))],"decomp_questions":[ip[i]['decomp_questions'] for i in range(len(ip))]}
    final_test_df= pd.DataFrame(final_test)  

    final_test_df.to_csv(f'src/qgen/infer/out/{case}.csv')

