# app.py
"""
Streamlit app that:
- Retrieves relevant chunks (RAG) using your retrieval.py
- Generates an answer via your LLM (via rag_pipeline.build_prompt + HF pipeline)
- Streams the answer text progressively in the UI (safe main-thread updates)
- Plays audio in parallel (server-side) while streaming text
"""

import streamlit as st
import queue
import threading
import time

from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

from src.retrieval import load_faiss_and_metadata, retrieve_chunks
from src.rag_pipeline import build_prompt
from src.post_processing import summarize_text, extract_action_points
from src.utils import clean_text, add_disclaimer
from src.tts import run_tts_stream

# -------------------------
# Config
# -------------------------
FAISS_INDEX_PATH = "data_processed/faiss_index.idx"
METADATA_PATH = "data_processed/chunks_metadata.pkl"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"

# how many chars of context to include (tune as needed)
CONTEXT_CHAR_LIMIT = 2000

# TTS chunk size (characters)
TTS_CHUNK_SIZE = 200

# -------------------------
# Load models (CPU)
# -------------------------
st.sidebar.info("Loading models — this may take a moment...")

# SentenceTransformer (avoid meta device issues by using direct constructor)
# embed_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
embed_model = SentenceTransformer(EMBEDDING_MODEL)
embed_model = embed_model.to("cpu")
# Load LLM explicitly (avoid lazy/meta loading)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL).to("cpu")

llm_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1
)

# Load FAISS & metadata
index, chunks = load_faiss_and_metadata(FAISS_INDEX_PATH, METADATA_PATH)

# -------------------------
# Streamlit UI
# -------------------------
st.title("Lexi — Hands-free Legal Assistant")
st.write("Ask questions about the Consumer Protection Act. Lexi will display and speak the answer in real-time.")

query = st.text_input("Enter your legal question:")

if st.button("Get Answer") and query.strip():
    with st.spinner("Thinking..."):
        # --- Retrieval
        retrieved = retrieve_chunks(query, index, chunks, embed_model, k=3)

        # Build a context string (join chunks) but truncate if too large
        context_joined = "\n\n".join([c["chunk"] for c in retrieved])
        if len(context_joined) > CONTEXT_CHAR_LIMIT:
            context_joined = context_joined[:CONTEXT_CHAR_LIMIT]

        # Build prompt using your rag_pipeline helper
        prompt = build_prompt(query, retrieved)  # build_prompt uses chunks, safe

        # LLM generation
        # reduce max tokens if model warns about large input
        result = llm_pipeline(prompt, max_new_tokens=250, do_sample=False)
        raw_answer = result[0].get("generated_text", "")

        # Clean answer but preserve paragraph breaks for readable bullets
        def clean_preserve_paragraphs(text: str) -> str:
            # convert CRLF to LF
            t = text.replace("\r\n", "\n").replace("\r", "\n")
            # collapse multiple empty lines to two newlines
            while "\n\n\n" in t:
                t = t.replace("\n\n\n", "\n\n")
            # strip trailing/leading whitespace
            return t.strip()

        answer = clean_preserve_paragraphs(clean_text(raw_answer))
        answer_with_disclaimer = add_disclaimer(answer)

        # Post-processing
        summary = summarize_text(answer)
        highlights = extract_action_points(answer)

        # ------------------------
        # STREAMING SETUP
        # ------------------------
        st.subheader("Full Answer")
        answer_placeholder = st.empty()

        # prepare a queue to receive progressive text from TTS thread
        out_q = queue.Queue()

        # Start background thread: it will push progressive text to out_q and speak
        t = threading.Thread(
            target=run_tts_stream,
            args=(answer_with_disclaimer, TTS_CHUNK_SIZE, out_q),
            daemon=True,
        )
        t.start()

        # Main thread: read from queue and update UI safely
        progressive = ""
        while True:
            try:
                item = out_q.get(timeout=5)  # wait up to 5s for new chunk
            except queue.Empty:
                # no new chunk for a while — stop waiting
                break

            if item is None:
                # finished
                break

            progressive = item  # progressive contains text up to current chunk end

            # display using markdown; preserve paragraphs and numbered/bullet lines
            # ensure blank lines respected by wrapping in triple quotes (markdown respects \n)
            answer_placeholder.markdown(progressive)

        # When finished streaming, ensure final full answer displayed
        answer_placeholder.markdown(answer_with_disclaimer)

        # Show summary and highlights (they appear after answer)
        if summary:
            st.subheader("Summary")
            st.write(summary)

        if highlights:
            st.subheader("📌 Actionable Highlights")
            for point in highlights:
                # ensure each point is on its own line
                st.markdown(f"- {point}")

        # st.success("Done speaking.")
