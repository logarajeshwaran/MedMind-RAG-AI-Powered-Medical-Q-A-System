import os
import pickle
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from config import (
    DATA_DIR, VECTORSTORE_DIR, EMBDEDDING_MODEL,
    FAISS_INDEX_PATH, FAISS_METADATA_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP
)


def load_medquad_data(filepath: str) -> list[dict]:
    """Load MedQuAD CSV and return list of QA dicts."""
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["answer"])
    df = df[["question", "answer"]].drop_duplicates()
    records = df.to_dict(orient="records")
    print(f"✅ Loaded {len(records)} QA pairs.")
    return records


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_documents(records: list[dict]) -> tuple[list[str], list[dict]]:
    """Convert QA records into chunks + metadata."""
    documents, metadatas = [], []

    for record in records:
        passage = f"Question: {record['question']}\nAnswer: {record['answer']}"
        chunks = chunk_text(passage)
        for chunk in chunks:
            documents.append(chunk)
            metadatas.append({
                "question": record["question"][:200],
                "source": "MedQuAD"
            })

    print(f"✅ Created {len(documents)} chunks.")
    return documents, metadatas


def ingest_to_faiss(documents: list[str], metadatas: list[dict]):
    """Embed documents and store in FAISS index."""
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    # Load embedding model
    print("🔄 Loading embedding model...")
    model = SentenceTransformer(EMBDEDDING_MODEL)

    # Generate embeddings in batches
    print("🔄 Generating embeddings...")
    embeddings = []
    batch_size = 256
    for i in tqdm(range(0, len(documents), batch_size), desc="Embedding batches"):
        batch = documents[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build FAISS index (Inner Product = cosine after normalization)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"✅ FAISS index saved → {FAISS_INDEX_PATH}")

    # Save metadata + documents as pickle
    with open(FAISS_METADATA_PATH, "wb") as f:
        pickle.dump({"documents": documents, "metadatas": metadatas}, f)
    print(f"✅ Metadata saved → {FAISS_METADATA_PATH}")


def run_ingestion(filename: str = "medquad.csv"):
    """Full pipeline: load → chunk → embed → store."""
    filepath = os.path.join(DATA_DIR, filename)
    records = load_medquad_data(filepath)
    documents, metadatas = build_documents(records)
    ingest_to_faiss(documents, metadatas)
    print("🎉 Ingestion complete! FAISS index is ready.")


if __name__ == "__main__":
    run_ingestion()