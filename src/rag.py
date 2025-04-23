import torch
import numpy as np
import chromadb
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from typing import List

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

class RAG:
    def __init__(self, database_path, collection_name, embedding_model="facebook/contriever", device=1):
        """
        Initialize the RAG system
        """       
        # Load embedding model
        self.device = device
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.embedding_model = AutoModel.from_pretrained(embedding_model)

        # Load the database
        client = chromadb.PersistentClient(path=database_path)
        self.collection = client.get_collection(collection_name)

    def get_embedding(self, text):
        """
        Get the embedding of a text
        """
        self.embedding_model.to(self.device)
        inputs = self.embedding_tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        embedding = mean_pooling(outputs[0], inputs['attention_mask']).cpu().detach().numpy()
        self.embedding_model.to("cpu")
        return embedding.squeeze().tolist()

    def retrieve(self, query, scope: List[int], top_k=3):
        """
        retrieve relevant passages based on semantic similarity.
        """
        query_embedding = self.get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"chapter": {"$in" : scope}}
        )
        return results['documents'][0]
    
if __name__ == "__main__":
    # Example usage
    rag = RAG(database_path='hp_vdbs/hp', collection_name='book')
    query = "What is the name of Harry Potter's pet owl? a) Hedwig b) Fawkes c) Crookshanks d) Scabbers"
    scope = [4, 5, 6]  # Example scope
    retrieved_contexts = rag.retrieve(query, scope, top_k=3)
    
    for context in retrieved_contexts:
        print(context)
        print("-----")