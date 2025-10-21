from typing import List
from transformers import AutoModel
from transformers import AutoTokenizer, ElectraForSequenceClassification
import numpy as np
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


class Scorer:
    def __init__(self) -> None:
        pass
        
    def score(self,doc:str,queries:List[str]):
        pass

class HfScorer(Scorer):

    def __init__(self,model="bert-large-uncased") -> None:
        super().__init__()
        self.device = torch.device("cuda:1")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)
        self.model.to(self.device)
        # self.scorer = BERTScorer(model_type=model)


    def score(self, doc, queries):
        inputs1 = self.tokenizer(doc, return_tensors="pt", padding=True, truncation=True)
        inputs2 = self.tokenizer(queries, return_tensors="pt", padding=True, truncation=True)

        inputs1 = {k: v.to(self.device) for k, v in inputs1.items()}
        inputs2 = {k: v.to(self.device) for k, v in inputs2.items()}
        outputs1 = self.model(**inputs1)
        outputs2 = self.model(**inputs2)
        embeddings1 = outputs1.last_hidden_state.mean(dim=1).detach().cpu().numpy()
        embeddings2 = outputs2.last_hidden_state.mean(dim=1).detach().cpu().numpy()
        norm_embeddings1 = embeddings1 / np.linalg.norm(embeddings1)
        norm_embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        similarity = np.dot(norm_embeddings1, norm_embeddings2.T)
        return similarity[0]
    
    # def score(self,claim,queries):
    #     _,_,F1 = self.scorer.score([claim]*len(queries),queries)
    #     return F1.tolist()

# class CosineScorer(Scorer):
        
#         def __init__(self,model="bert-base-uncased") -> None:
#         super().__init__()
#         # self.tokenizer = BertTokenizer.from_pretrained(model)
#         # self.model = BertModel.from_pretrained(model)
#         self.scorer = BERTScorer(model_type='bert-base-uncased')





class ElectraScorer(Scorer):

    def __init__(self,tokeinizer='google/electra-base-discriminator', model='crystina-z/monoELECTRA_LCE_nneg31') -> None:
        super().__init__()
        self.tokeniser = AutoTokenizer.from_pretrained(tokeinizer)
        self.model = ElectraForSequenceClassification.from_pretrained(model).eval()
            
    
    def score(self,doc,queries):
        passages = [doc]*len(queries)
        inps = self.tokeniser(passages, queries, return_tensors='pt', padding=True, truncation=True)
        output = self.model(**inps).logits

        return output

        




    
class RelevanceFilter:
    def __init__(self,scorer:Scorer,threshold:float,alpha:float) -> None:
        self.scorer = scorer
        self.alpha = alpha
        self.threshold = threshold
        self.exclude_words = ["fact-check","veracity","in the claim","PolitiFact"]

    def filter(self,doc,queries,claim=None,k=None):

        filtered_queries = []
        for query in queries:
            include = True
            for word in self.exclude_words:
                if(word.lower() in query.lower()):
                    include=False
                    break
            if(include):
                filtered_queries.append(query)
        if(not(filtered_queries)):
            return []

        doc_similarities= self.scorer.score(doc,filtered_queries)
        if(claim):
            claim_similarities = self.scorer.score(claim,filtered_queries)
            final_relevance = [i*self.alpha+(1-self.alpha)*j for i,j in zip(claim_similarities,doc_similarities)]
        else:
            final_relevance = doc_similarities
        filtered_queries_new = []

        

        for relevance,query in zip(final_relevance,filtered_queries):
            if(relevance>self.threshold):
                filtered_queries_new.append((relevance,query))
        filtered_queries_new.sort(key=lambda x:x[0],reverse=True)

        k = len(filtered_queries) if not(k) else min(len(filtered_queries),k)
        
        return [x[1] for x in filtered_queries_new][:k]



class MMR(Scorer):
    def __init__(self,model_name=None,beta=0.2):
        self.model = SentenceTransformer(model_name,device='cuda:1')
        self.beta = beta

    def standardize_normalize_cosine_similarities(self, cosine_similarities):
        """Normalized cosine similarities"""
        # normalize into 0-1 range
        cosine_sims_norm = (cosine_similarities - np.min(cosine_similarities)) / (
                np.max(cosine_similarities) - np.min(cosine_similarities))

        # standardize and shift by 0.5
        cosine_sims_norm = 0.5 + (cosine_sims_norm - np.mean(cosine_sims_norm)) / np.std(cosine_sims_norm)

        return cosine_sims_norm       

    def get_embeddings_for_data(self, data_ls):
        
        embeddings = self.model.encode(data_ls)
        return embeddings
    
    def max_normalize_cosine_similarities_pairwise(self,cosine_similarities):
        """Normalized cosine similarities of pairs which is 2d matrix of pairwise cosine similarities"""
        cosine_sims_norm = np.copy(cosine_similarities)
        np.fill_diagonal(cosine_sims_norm, np.NaN)

        # normalize into 0-1 range
        cosine_sims_norm = (cosine_similarities - np.nanmin(cosine_similarities, axis=0)) / (
                np.nanmax(cosine_similarities, axis=0) - np.nanmin(cosine_similarities, axis=0))

        # standardize shift by 0.5
        cosine_sims_norm = \
            0.5 + (cosine_sims_norm - np.nanmean(cosine_sims_norm, axis=0)) / np.nanstd(cosine_sims_norm, axis=0)

        return cosine_sims_norm

    def max_normalize_cosine_similarities(self, cosine_similarities):
        """Normalize cosine similarities using max normalization approach"""
        return 1 / np.max(cosine_similarities) * cosine_similarities.squeeze(axis=1)

    def score(self, sentence, data,k=None):
        k = len(data)
        sent_emb = self.get_embeddings_for_data(sentence)
        data_emb = self.get_embeddings_for_data(data)
        text_sims = cosine_similarity(data_emb, [sent_emb]).tolist()
        candidate_sims = cosine_similarity(data_emb)
        text_sims_norm = self.standardize_normalize_cosine_similarities(text_sims)
        phrase_sims_norm = self.max_normalize_cosine_similarities_pairwise(candidate_sims)

        selected_data_indices = []
        selected_data_scores = []
        unselected_data_indices = list(range(len(data)))

        # find the most similar doc (using original cosine similarities)
        best_idx = np.argmax(text_sims)
        best_val = np.max(text_sims)
        selected_data_indices.append(best_idx)
        selected_data_scores.append(best_val)
        unselected_data_indices.remove(best_idx)

        # do top_n - 1 cycle to select top N data
        for _ in range(min(len(data), k) - 1):
            unselected_data_distances_to_text = text_sims_norm[unselected_data_indices, :]
            unselected_data_distances_pairwise = phrase_sims_norm[unselected_data_indices][:,
                                                   selected_data_indices]

            # if dimension of data distances is 1 we add additional axis to the end
            if unselected_data_distances_pairwise.ndim == 1:
                unselected_data_distances_pairwise = np.expand_dims(unselected_data_distances_pairwise, axis=1)
            
            relevances = self.beta * unselected_data_distances_to_text - (1 - self.beta) * np.max(unselected_data_distances_pairwise,
                                                                                 axis=1).reshape(-1, 1)
            
            next_indx,score =  np.argmax(relevances),np.max(relevances)
            # find new candidate with
            idx = int(next_indx)
            best_idx = unselected_data_indices[idx]

            # select new best docs and update selected/unselected phrase indices list
            selected_data_indices.append(best_idx)
            selected_data_scores.append(score)
            unselected_data_indices.remove(best_idx)
        selected_data_scores = np.array(selected_data_scores)
        my_min_val = np.min(selected_data_scores)
        my_max_val = np.max(selected_data_scores)

        # Perform min-max normalization
        normalized_scores = (selected_data_scores - my_min_val) / (my_max_val - my_min_val)
        normalized_scores = normalized_scores.tolist()
        
        top_sent_scores = []
        for idx in range(k):
            score = normalized_scores[selected_data_indices.index(idx)]
            top_sent_scores.append(score)


        return top_sent_scores

import json
from typing import Any, Dict
from urllib.parse import urlparse


class FactCheckFilter:

    def __init__(self,fact_check_path) -> None:
        self.fact_checkers = self._get_fact_checker_by_domain(fact_check_path)

    def get_domain_name(self,url: str) -> str:
        """Parse a domain name out of an URL.

        Args:
            url: URL to parse.

        Returns:
            Domain name.
        """
        try:
            # Parse the URL
            parsed_url = urlparse(url)

            # Get the netloc part of the URL
            domain_name = parsed_url.netloc

            # If the URL contains 'www.', remove it to get the clean domain name
            if "www." in domain_name:
                domain_name = domain_name.replace("www.", "")

            return domain_name

        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
    def _get_fact_checker_by_domain(self,path):
        """Loads the fact checkers from the JSON file.

        Returns:
            A dictionary of fact checkers by domain name.
        """
        fact_checkers = self._read_json(path)
        fact_checkers_by_domain = {}
        for fact_checker in fact_checkers:
            fact_checkers_by_domain[fact_checker["query_name"]] = fact_checker
        return fact_checkers_by_domain
        
    def is_not_fact_checker(
        self, fact_url: str, evidence_url: str) -> bool:
        """Check if the evidence URL is a not fact checker.

        Args:
            fact_checkers: Dict of fact checkers by domain name.
            fact_url: URL of the fact or claim.
            evidence_url: URL of the evidence obtained from Factiverse API.

        Returns:
            True if the evidence URL is a fact checker, False otherwise.
        """
        return (
            evidence_url != fact_url
            and self.get_domain_name(evidence_url) not in self.fact_checkers
            and "fact_check" not in evidence_url
            and "fact-check" not in evidence_url
            and "factchecks" not in evidence_url.lower()
            and "factcheck" not in evidence_url.lower()
            and "claimbuster" not in evidence_url.lower()
            and "politifact" not in self.get_domain_name(evidence_url.lower())
            and "github" not in evidence_url.lower()
        )
    
    def _read_json(self,path) -> Dict[Any, Any]:
        """Reads the fact check dataset from the given path.

        Args:
            path: The path to the dataset.

        Returns:
            A list of facts.
        """
        with open(path) as f:
            data = json.load(f)
        return data

def get_fact_checker_by_domain(self,path):
    """Loads the fact checkers from the JSON file.

    Returns:
        A dictionary of fact checkers by domain name.
    """
    fact_checkers = self.read_json(path)
    fact_checkers_by_domain = {}
    for fact_checker in fact_checkers:
        fact_checkers_by_domain[fact_checker["query_name"]] = fact_checker
    return fact_checkers_by_domain







