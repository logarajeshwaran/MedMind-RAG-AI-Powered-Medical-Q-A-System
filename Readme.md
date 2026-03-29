# 🩺 MedMind RAG

A Retrieval-Augmented Generation (RAG) system for medical question answering, built on the [MedQuAD](https://github.com/abachaa/MedQuAD) dataset. Ask natural-language medical questions and get grounded, evidence-backed answers powered by FAISS vector search and GPT-4o.

---

## Features

- Semantic search over the MedQuAD dataset using FAISS + `all-MiniLM-L12-v2` embeddings
- GPT-4o answer synthesis grounded strictly in retrieved context
- Streamlit chat UI with source viewer
- FastAPI REST API with Pydantic-validated request/response models
- Modular pipeline: ingest → retrieve → generate

---

## Project Structure

```
medmind-rag/
├── api/
│   └── main.py              # FastAPI app (POST /ask, GET /health)
├── app/
│   └── streamlit_app.py     # Streamlit chat interface
├── data/
│   └── medquad.csv          # MedQuAD source dataset
├── scripts/
│   └── run_ingest.py        # Entry point to build the FAISS index
├── src/
│   ├── config.py            # Paths, model names, hyperparameters
│   ├── ingest.py            # Load → chunk → embed → store pipeline
│   ├── reteriver.py         # FAISS similarity search
│   └── rag_chain.py         # Prompt builder + OpenAI completion
├── vectorstore/             # Generated FAISS index & metadata (git-ignored)
├── .env                     # API keys (not committed)
└── requirements.txt
```

---

## Quickstart

### 1. Clone & install dependencies

```bash
git clone <repo-url>
cd medmind-rag
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
```

### 3. Build the FAISS vector index

```bash
python scripts/run_ingest.py
```

This reads `data/medquad.csv`, chunks the QA pairs, embeds them with `all-MiniLM-L12-v2`, and saves the FAISS index to `vectorstore/`.

### 4. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

### 4. Run the FastAPI server (optional)

```bash
uvicorn api.main:app --reload
```

API docs available at `http://localhost:8000/docs`.

---

## API Reference

### `POST /ask`

```json
// Request
{ "question": "What are the symptoms of diabetes?" }

// Response
{
  "query": "What are the symptoms of diabetes?",
  "answer": "...",
  "sources": [
    {
      "text": "...",
      "metadata": { "question": "...", "source": "MedQuAD" },
      "distance": 0.1234
    }
  ]
}
```

### `GET /health`

```json
{ "status": "ok", "service": "Medical RAG API running...." }
```

---

## Configuration

All tunable parameters live in [src/config.py](src/config.py):

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L12-v2` | Sentence transformer for embeddings |
| `LLM` | `gpt-4o` | OpenAI model for answer generation |
| `MAX_TOKENS` | `1024` | Max tokens in the LLM response |
| `TOP_K_RESULT` | `5` | Number of passages retrieved per query |
| `CHUNK_SIZE` | `500` | Characters per text chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L12-v2`) |
| Vector store | FAISS (`IndexFlatIP` with L2 normalisation) |
| LLM | OpenAI GPT-4o |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Data | MedQuAD (47k+ medical QA pairs) |

---

## Disclaimer

This tool is for **educational and research purposes only**. It is **not** a substitute for professional medical advice, diagnosis, or treatment.
