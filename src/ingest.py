import os
import pickle
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from  import SentenceTransformer
from config import ( DATA_DIR , VECTORSTORE_DIR , EMBDEDDING_MODEL , FAISS_INDEX_PATH, FAISS_METADATA_PATH,
CHUNK_SIZE
,CHUNK_OVERLAP )


def load_medquad_data(filepath):
    df = pd.read_csv(filepath,headers=None)
    df.columns=["question","answer"]
    df = df[["question","answer"]]
    df = df.dropna(subset=["answer"])
    df = df[["question","answer"]].drop_duplicates()
    records = df.to_dict(orient="records")
    print(f" Loaded {len(records)} QA Pairs")
    return records


def chunk_text(text,chunk_size=CHUNK_SIZE,overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def build_document(records):
    documents , metadata = [] , []
    for record in records:
        passage = f" Question : {record["question"]} \n Answer : {record["answer"]}"
        chunks = chunk_text([passage])
        for chunk in chunks:
            documents.append(chunk)
            metadata.append({
                "quesion": record["question"][:200],
                "source" : "medqa"
            })

    print(f" Created {len(documents)} Chunks")
    return documents , metadata

def ingest_to_faiss(documents,metadata):
    os.makedir(VECTORSTORE_DIR,exists_ok=True)

    print("Loading Embedding Model ......")

    model = SentenceTransformer(EMBDEDDING_MODEL)

    print(" Generating Embedding .....")

    embedding = []
    batch_size = 256

    for i in tqdm(range(0,len(documents),batch_size),desc="Embedding Batches"):
        batch = documents[i:i+batch_size]
        batch_embedding = model.encode(batch,show_progress_bar=False)
        embedding.append(batch_embedding)

    embeddings = np.vstack(embedding).astype("float32")

    faiss.normalize_L2(embeddings)

    dimensions = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimensions)
    index.add(embeddings)

    faiss.write_index(index,FAISS_INDEX_PATH)


    with open(FAISS_METADATA_PATH,"wb") as f:
        pickle.dump({"documents" : documents , "metadata" : metadata}, f)


def run_ingestion(file_name):
    filepath = os.path.join(DATA_DIR,file_name)
    records = load_medquad_data(filepath)
    documents , metadata        = build_document(records)
    ingest_to_faiss(documents , metadata)

if __name__ == "__main__":
    run_ingestion()