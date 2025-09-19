"""
rag_pipeline.py
---------------
Phase 3: RAG (Retrieval-Augmented Generation)

This script:
1. Accepts a user query
2. Retrieves top-k chunks (via retrieval.py)
3. Builds a context prompt for the LLM
4. Generates a natural answer with citations
"""

import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from src.retrieval import load_faiss_and_metadata, retrieve_chunks

# ------------------------------
# Configuration
# ------------------------------
FAISS_INDEX_PATH = "../data_processed/faiss_index.idx"
METADATA_PATH = "../data_processed/chunks_metadata.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"   # Lightweight, good for testing

# ------------------------------
# Core Functions
# ------------------------------

def build_prompt(query: str, retrieved_chunks: list) -> str:
    """
    Build a clean context prompt for the LLM without file references.
    """
    context_text = "\n\n".join([c['chunk'] for c in retrieved_chunks])
    prompt = (
        f"Answer the following question based only on the context provided.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        f"Answer clearly in plain text, without bullet points, without headings, "
        f"and without repeating context."
    )
    return prompt

def generate_answer(query: str, index, chunks, embed_model, llm_pipeline, k: int = 5) -> str:
    """
    Main RAG function: retrieve context + run through LLM.
    """
    # Step 1: Retrieve chunks
    retrieved = retrieve_chunks(query, index, chunks, embed_model, k=k)

    # Step 2: Build LLM prompt
    prompt = build_prompt(query, retrieved)

    # Step 3: Generate answer
    result = llm_pipeline(prompt, max_new_tokens=300, do_sample=False)
    return result[0]['generated_text']


# ------------------------------
# Run a test when executed directly
# ------------------------------
if __name__ == "__main__":
    print("=== Phase 3: RAG Pipeline Test ===")

    # Load models
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    llm_pipeline = pipeline("text2text-generation", model=LLM_MODEL, device=-1)  # CPU

    # Load FAISS + metadata
    index, chunks = load_faiss_and_metadata(FAISS_INDEX_PATH, METADATA_PATH)

    # Sample query
    query = "What remedies are available to consumers under the Consumer Protection Act?"

    # Generate answer
    answer = generate_answer(query, index, chunks, embed_model, llm_pipeline, k=3)

    print("\n=== FINAL ANSWER ===")
    print(answer)

    print("\n[INFO] Phase 3 test completed ")
