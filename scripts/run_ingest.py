import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import run_ingestion

if __name__ == '__main__':
    run_ingestion(filename="")