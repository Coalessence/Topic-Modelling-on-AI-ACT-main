from hdbscan import HDBSCAN
import pandas as pd
from sklearn.cluster import KMeans
from bertopic import BERTopic
from transformers.pipelines import pipeline
import numpy as np
import torch

class BertEx:
    
    def __init__(self, data=None):
        self.legalbert_model = pipeline("feature-extraction", model="nlpaueb/legal-bert-small-uncased", use_auth_token=True, device="cuda" if torch.cuda.is_available() else "cpu")
        hdbscan_model = HDBSCAN(min_cluster_size=26, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        kmeans_model = KMeans(n_clusters=10)
        self.topic_model = BERTopic(hdbscan_model=kmeans_model)
        self.model_fit(data) if data is not None else None

    def pipeline_avg_for_legal_bert(self, texts):
        legalbert_model = pipeline("feature-extraction", model="nlpaueb/legal-bert-small-uncased", device="cuda" if torch.cuda.is_available() else "cpu", use_auth_token=True)
        new_arr = np.zeros((len(texts), 512))
        embeddings = legalbert_model(texts)
        
        for i, document in enumerate(embeddings) :
            document = document[0]
            sum = 0.0
            
            for j in range(0, len(document)): 
                sum += np.array(document[j])
            
            new_arr[i,:] = sum / len(document) 
            
        return new_arr

    def model_fit(self, data):
        self.topics, self.probs = self.topic_model.fit_transform(data)
    
    def get_tab_data(self):
        return self.topic_model.get_topic_info()
    
    def get_graph_data(self, data=None):
        return self.topic_model.visualize_documents(data)

