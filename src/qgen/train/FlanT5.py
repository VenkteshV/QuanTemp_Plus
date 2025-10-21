
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


# import pytorch_lightning as pl

# from pytorch_lightning.callbacks import ModelCheckpoint


import random

import datetime


import datasets
from IPython.display import display, HTML
import random
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import Dataset, load_metric

import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback
)
import json

from utils.dir_utils import create_fir_if_na

with open('data/test.json') as fp:
    claims_test = json.load(fp)

with open('data/train.json') as fp:
    claims_train = json.load(fp)

with open('data/val.json') as fp:
    claims_val = json.load(fp)

nltk.download('punkt')
import os


import wandb
wandb.login()


"""#### References: [Official](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/summarization.ipynb#scrollTo=vc0BSBLIIrJQ), [Community](https://colab.research.google.com/github/elsanns/xai-nlp-notebooks/blob/master/fine_tune_bart_summarization_two_langs.ipynb#scrollTo=6R9d7ELIpX9F)"""



MODEL_PATH = 'google/flan-t5-large'
TOKENIZER_PATH = 'google/flan-t5-large'
SAVE_PATH = "dataset_experiments/snippets/qgen/train/out/bart/"
SAVE_MODEL_PATH = "dataset_experiments/snippets/qgen/train/models/bart/"
LOGGING_PATH = "dataset_experiments/snippets/qgen/train/logs/"
create_fir_if_na(SAVE_PATH)
create_fir_if_na(SAVE_MODEL_PATH)



"""# Loading and processing the data"""

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

metric = load_metric("rouge")

raw_datasets = {"train":train_dataset,"test":test_dataset,"val":val_dataset}

def show_random_elements(dataset, num_examples=5):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))

show_random_elements(raw_datasets["train"])

"""## Tokenization and Dataset Preparation"""

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

tokenizer

max_input_length = 64
max_target_length = 128

def preprocess_function(examples, prefix="decompose the compositional claim into queries:"):
    inputs = [prefix + doc for doc in examples['claim']]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    outputs = ["[SEP]".join(q) for q in examples['queries']]

    labels = tokenizer(text_target=outputs, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=raw_datasets['train'].column_names)
tokenized_test = test_dataset.map(preprocess_function, batched=True, remove_columns=raw_datasets['train'].column_names)
tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=raw_datasets['train'].column_names)



"""# Building the model

## Metrics
"""

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    print(predictions[0])
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    print("decoded gen",decoded_preds)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

"""## Loading the model"""

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)



"""## Training Args"""

epochs = 6
batch_size = 8
lr = 2e-5

early_stop = EarlyStoppingCallback(2, 1e-4)

args = Seq2SeqTrainingArguments(
    output_dir=SAVE_PATH,
    learning_rate=lr,
    do_train = True,
    do_eval = True,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    load_best_model_at_end=True,
    num_train_epochs=epochs,
    predict_with_generate=True,
    generation_max_length=512,
    logging_dir=LOGGING_PATH,
    logging_steps=300,
    save_steps=300,
    report_to = "wandb",
)

wandb_run = wandb.init(
    project="bart_subq",
    config={
        "per_device_train_batch_size": batch_size,
        "learning_rate": lr})

now = datetime.datetime.now()
current_time = now.strftime("%m/%d/%Y, %H:%M:%S")
wandb_run.name = "run_" + current_time

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[early_stop])

"""# Training"""

res = trainer.evaluate() # Evaluation before fine-tuning



# %%wandb

trainer.train()

trainer.evaluate()

trainer.save_model(SAVE_MODEL_PATH)

wandb_run.finish()

