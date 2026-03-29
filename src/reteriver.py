import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.config import (FAISS_INDEX_PATH,FAISS_METADATA_PATH,EMBDEDDING_MODEL,TOP_K_RESULT)


def load_vector_data():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_METADATA_PATH,'rb') as f:
        store = pickle.load(f)
    return index , store['documents'] , store['metadatas']


def retrieve(query,top_k=TOP_K_RESULT):
    index , documents , metadatas  = load_vector_data()
    model = SentenceTransformer(EMBDEDDING_MODEL)
    query_embedding = model.encode([query]).astype("float32")

    faiss.normalize_L2(query_embedding)

    score ,indices =  index.search(query_embedding,top_k)

    reterived = []

    for score , idx in zip(score[0],indices[0]):
        if idx == -1:
            continue
        reterived.append({
            "text" : documents[idx],
            "metadata" : metadatas[idx],
            "distance" : round(float(1 - score),4) 
        })
    
    return reterived