import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR,"data")
VECTORSTORE_DIR = os.path.join(BASE_DIR,"vectorstore")


EMBDEDDING_MODEL = "all-MiniLM-L12-v2"

FAISS_INDEX_PATH = os.path.join(VECTORSTORE_DIR,"faiss.index")
FAISS_METADATA_PATH = os.path.join(VECTORSTORE_DIR,"metadata.pkl")


LLM = "gpt-4o"
MAX_TOKENS = 1024

TOP_K_RESULT = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50