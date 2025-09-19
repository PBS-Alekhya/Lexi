# phase1_preprocessing_mixedlang.py

import os
import pdfplumber
from langdetect import detect
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
from tqdm import tqdm

# ---------------------------
# Configuration
# ---------------------------
PDF_FOLDER = "../data/"           # Folder with PDFs
FAISS_INDEX_PATH = "../data_processed/faiss_index.idx"
METADATA_PATH = "../data_processed/chunks_metadata.pkl"
CHUNK_SIZE = 300                    # words per chunk
OVERLAP = 50                        # word overlap between chunks
MIN_WORDS_PARAGRAPH = 10            # Minimum words to consider for language detection

# ---------------------------
# Step 1: Extract Text from PDFs
# ---------------------------
def extract_text_from_pdfs(pdf_folder):
    pdf_texts = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text:
                        pdf_texts.append({
                            "pdf_name": pdf_file,
                            "page": page_num,
                            "text": text.strip()
                        })
    return pdf_texts

print("Extracting text from PDFs...")
pdf_texts = extract_text_from_pdfs(PDF_FOLDER)
print(f"Total pages extracted: {len(pdf_texts)}")

# ---------------------------
# Step 2: Paragraph-level English Filtering
# ---------------------------
def filter_english_text(text, min_words=MIN_WORDS_PARAGRAPH):
    """
    Split text into paragraphs and keep only English paragraphs.
    """
    paragraphs = text.split("\n")
    english_paragraphs = []
    for para in paragraphs:
        if len(para.split()) < min_words:
            continue
        try:
            if detect(para) == "en":
                english_paragraphs.append(para)
        except:
            continue
    return " ".join(english_paragraphs)

english_pages = []
for page in pdf_texts:
    filtered_text = filter_english_text(page["text"])
    if filtered_text:
        english_pages.append({
            "pdf_name": page['pdf_name'],
            "page": page['page'],
            "text": filtered_text
        })

print(f"English pages kept after paragraph-level filtering: {len(english_pages)}")

# ---------------------------
# Step 3: Chunk Text
# ---------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

all_chunks = []
for page in english_pages:
    chunks = chunk_text(page['text'])
    for chunk in chunks:
        all_chunks.append({
            "pdf_name": page['pdf_name'],
            "page": page['page'],
            "chunk": chunk
        })

print(f"Total chunks created: {len(all_chunks)}")

# ---------------------------
# Step 4: Generate Embeddings
# ---------------------------
print("Generating embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [c["chunk"] for c in all_chunks]
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings)
print(f"Embeddings shape: {embeddings.shape}")

# ---------------------------
# Step 5: Build FAISS Index
# ---------------------------
print("Building FAISS index...")
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)
faiss.write_index(index, FAISS_INDEX_PATH)

# Save chunk metadata
with open(METADATA_PATH, "wb") as f:
    pickle.dump(all_chunks, f)

print(f"FAISS index saved to: {FAISS_INDEX_PATH}")
print(f"Chunk metadata saved to: {METADATA_PATH}")
print("Phase 1 preprocessing complete!")
