"""
retrieval.py
-------------
Phase 2: Retrieval Layer

This script helps in:
1. Loading FAISS index and metadata
2. Embedding user queries
3. Retrieving top-k relevant chunks from legal PDFs
4. Testing retrieval with a sample query
"""

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# ------------------------------
# Configuration
# ------------------------------
FAISS_INDEX_PATH = "../data_processed/faiss_index.idx"
METADATA_PATH = "../data_processed/chunks_metadata.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight, CPU-friendly

# ------------------------------
# Core Functions
# ------------------------------

def load_faiss_and_metadata(index_path: str, metadata_path: str):
    """
    Load the FAISS index and metadata (chunks information).
    """
    print("[INFO] Loading FAISS index and metadata...")
    index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"[INFO] Loaded FAISS with {index.ntotal} vectors.")
    print(f"[INFO] Loaded metadata with {len(chunks)} chunks.")
    return index, chunks


def embed_query(query: str, model) -> np.ndarray:
    """
    Convert a user query into an embedding.
    """
    return model.encode([query])


def retrieve_chunks(query: str, index, chunks, model, k: int = 5):
    """
    Retrieve top-k relevant chunks for a given query.
    Returns: list of dicts containing PDF name, page, and chunk text.
    """
    print(f"\n[INFO] Retrieving top {k} chunks for query: '{query}'")

    query_emb = embed_query(query, model)
    D, I = index.search(np.array(query_emb), k)

    results = []
    for idx in I[0]:
        results.append({
            "pdf_name": chunks[idx]['pdf_name'],
            "page": chunks[idx]['page'],
            "chunk": chunks[idx]['chunk']
        })
    return results


def format_results(results):
    """
    Nicely print the retrieved chunks.
    """
    print("\n[RESULTS]")
    for i, r in enumerate(results, start=1):
        print(f"\nResult {i} - {r['pdf_name']} (Page {r['page']})")
        print(r['chunk'][:300], "...")


# ------------------------------
# Run a test when executed directly
# ------------------------------
if __name__ == "__main__":
    print("=== Phase 2: Retrieval Layer Test ===")

    # Load model once
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Load FAISS and metadata
    index, chunks = load_faiss_and_metadata(FAISS_INDEX_PATH, METADATA_PATH)

    # Test with a sample query (you can change this)
    sample_query = "What are the rights of a consumer under the Consumer Protection Act?"
    results = retrieve_chunks(sample_query, index, chunks, model, k=3)

    # Display results
    format_results(results)

    print("\n[INFO] Retrieval test completed successfully ✅")
