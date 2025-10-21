from typing import Dict, List
import torch
from transformers import AutoTokenizer
from contriever.src.contriever import Contriever
import numpy as np
import faiss
import tqdm
from dateutil.parser import parse



import heapq
from transformers.models.auto.modeling_auto import AutoModel

class MaxHeap:
    def __init__(self, k):
        self.k = k
        self.heap = []

    def push(self, item):
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (-item[0],item[1]))  # Pushing negated value for max heap
        else:
            if item[0] > -self.heap[0][0]:
                heapq.heappop(self.heap)
                heapq.heappush(self.heap, (-item[0],item[1]))

    def get_top_k(self):
        return sorted([(-x[0],x[1]) for x in self.heap],key=lambda x: -x[0])

class BiEncoderRetriever():
    def __init__(self,corpus_model_name:str='',device:str='cuda:1',query_model_name=None) -> None:
        self.device = torch.device(device)
        self.load_model(corpus_model_name,query_model_name)
        self.index=None

    def load_model(self,corpus_encoder,query_encoder):
        self.tokenizer = AutoTokenizer.from_pretrained(corpus_encoder)
        self.c_model = AutoModel.from_pretrained(corpus_encoder)
        self.c_model.to(self.device)
        if(query_encoder):
            self.q_model = AutoModel.from_pretrained(query_encoder)
        else:
            self.q_model = self.c_model
        self.c_model.eval()
        self.q_model.eval()
    
    def index_corpus(self,input:List[str],load_path:str=None):
        if(load_path):
            cpu_index = faiss.read_index(load_path)
            res = faiss.StandardGpuResources() 
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            corpus_embeddings = self.encode_norm(input,is_query=False)
            res = faiss.StandardGpuResources() 
            index_flat = faiss.IndexFlatIP(corpus_embeddings.shape[1])

            gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
            gpu_index_flat.add(corpus_embeddings) 
            self.index = gpu_index_flat

    def save_index(self,save_path):
        cpu_index = faiss.index_gpu_to_cpu(self.index)
        faiss.write_index(cpu_index, save_path)

    
    
    def encode(self,input:List[str],batch_size:int=32,is_query=False):
        embeddings, batch = [], []
        with torch.no_grad():

            for k, q in enumerate(input):
                batch.append(q)

                if len(batch) == batch_size or k == len(input) - 1:
                    output = self.encode_batch(batch,is_query)
                    if(output.device!='cpu'):
                        embeddings.append(output.cpu())
                    else:
                        embeddings.append(output.cpu())
                    batch = []

        embeddings = torch.cat(embeddings, dim=0)
        print(f"Questions embeddings shape: {embeddings.size()}")

        return embeddings.numpy()

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    
    def encode_batch(self,input,is_query=False):
        with torch.no_grad():
            tokenized_questions = self.tokenizer(input, padding=True, truncation=True, return_tensors='pt').to(self.device)
            if(is_query):
                token_emb =  self.q_model(**tokenized_questions)
            else:
                token_emb =  self.c_model(**tokenized_questions)
        sentence_emb = self.mean_pooling(token_emb[0],tokenized_questions["attention_mask"])
        return sentence_emb


    def encode_norm(self,input:List[str],batch_size:int=32,is_query=False):
        embeddings = self.encode(input,batch_size,is_query)
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    def normalize_dict_values(data):
        if(len(data)<=1):
            return data
        # Find the maximum and minimum values in the dictionary
        max_val = max(data.values())
        min_val = min(data.values())
        
        # Normalize each value in the dictionary
        normalized_data = {key: (value - min_val) / (max_val - min_val) for key, value in data.items()}
        
        return normalized_data
    
    
    def retrieve(self,input:List[str],k,threshold:float):
        retrieval_results ={}
        empty_results = 0

        query_embeddings = self.encode_norm(input,is_query=True)

        D, I = self.index.search(query_embeddings, k)

        for i in range(len(input)):
            retrieval_results[str(i)] = {}
            for score,passageId in zip(D[i],I[i]):
                if(score>threshold):
                    retrieval_results[str(i)][str(passageId)] = score.item()
            if(not(retrieval_results[str(i)])):
                empty_results+=1
                
        print(f"{empty_results}/{len(input)} queries had no results fetched for them")

        return retrieval_results

    def retrieve_combmax(self,queries,threshold=0.5):
        qqrel = {}

        if(not(queries)):
            qqrel ={}
        else:
            retrieval_results = self.retrieve(queries,2048,0.5)
            doc_scoring = {}
            for result in retrieval_results:
                scores = self.normalize_dict_values(retrieval_results[result])
                for doc_id in scores:
                    if(doc_id in doc_scoring):
                        doc_scoring[doc_id] = max(scores[doc_id],doc_scoring[doc_id])
                    else:
                        doc_scoring[doc_id] = scores[doc_id]

            sorted_documents = sorted(doc_scoring.items(), key=lambda x: x[1], reverse=True)
            for doc_id,score in sorted_documents:
                qqrel[str(doc_id)] = score
        return qqrel
    
class BiEncoderRetrieverTemporal(BiEncoderRetriever):

    def __init__(self,corpus_model_name:str='',device:str='cuda:1',query_model_name=None) -> None:
        super().__init__(corpus_model_name=corpus_model_name,device=device,query_model_name=query_model_name)
        self.default_date =  parse('01-01-0001')

    def index_corpus(self,input:List,load_path:str=None):
        corpus,self.corpus_date_map = self.__create_date_map(input)
        super().index_corpus(corpus,load_path)
    
    def __create_date_map(self,input_dict):
        date_map = {}
        idx = 0
        input = []
        for text,date in input_dict:
            input.append(text)
            try:
                date_map[idx] = parse(date)
            except:
                date_map[idx] = self.default_date
            idx+=1
        return input,date_map

    def date_filter(self,D,I,date_map,k):
        top_k_all =[]
        for i in range(len(I)):
            top_k = []
            j = 0
            c = 0
            while(j<k and c<len(I[i])):
                if(self.corpus_date_map[I[i][c]]<date_map[i]):
                    top_k.append((D[i][c],I[i][c]))
                    j+=1
                c+=1
            top_k_all.append(top_k)
        return top_k_all      
            
        
    
    def retrieve(self,input:Dict[str,str],k,threshold:float,verbose=False):
        retrieval_results ={}
        empty_results = 0
        input,date_map = self.__create_date_map(input)


        query_embeddings = self.encode_norm(input)

        D, I = self.index.search(query_embeddings, 2048)
        top_k = self.date_filter(D,I,date_map,k)


        for i in range(len(input)):
            retrieval_results[str(i)] = {}
            for score,passageId in top_k[i]:
                if(score>=threshold):
                    retrieval_results[str(i)][str(passageId)] = score.item()
            if(not(retrieval_results[str(i)])):
                empty_results+=1
        if(verbose):
            print(f"{empty_results}/{len(input)} queries had no results fetched for them")

        return retrieval_results


    
class BiEncoderContriever(BiEncoderRetriever):
    def __init__(self,corpus_model_name:str='',device:str='cuda:1',query_model_name=None) -> None:
        super().__init__(corpus_model_name=corpus_model_name,device=device,query_model_name=query_model_name)
    
    def load_model(self,corpus_model_name,query_model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(corpus_model_name)
        self.c_model = Contriever.from_pretrained(corpus_model_name)
        self.c_model.to(self.device)
        if(query_model_name):
            self.q_model = Contriever.from_pretrained(corpus_model_name)
            self.q_model.to(self.device)
        else:
            self.q_model = self.c_model

    def encode_batch(self,input,is_query=False):
        encoded_batch = self.tokenizer.batch_encode_plus(
            input,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded_batch = {k: v.to(device=self.device) for k, v in encoded_batch.items()}
        if(is_query):
            return self.q_model(**encoded_batch)
        else:
            return self.c_model(**encoded_batch)


class BiEncoderContrieverTemporal(BiEncoderContriever,BiEncoderRetrieverTemporal):

    def __init__(self,corpus_model_name:str='',device:str='cuda:1',query_model_name=None) -> None:
        super().__init__(corpus_model_name=corpus_model_name,device=device,query_model_name=query_model_name)
        self.default_date =  parse('01-01-0001')





         


