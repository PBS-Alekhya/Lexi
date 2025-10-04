# Lexi - Voice-Powered Legal Assistant

A voice-powered AI legal assistant that combines Retrieval-Augmented Generation (RAG) with speech capabilities.
This project allows users to ask legal questions via voice or text, retrieves relevant knowledge chunks, and generates natural answers with citations. Answers can be read out using Text-to-Speech (TTS) for a seamless experience.

# 🚀 Features

Voice & Text Input – interact with the assistant using either text or speech.

Retrieval-Augmented Generation (RAG) – fetches top-k relevant chunks from legal documents.

LLM-based Answering – uses Hugging Face models to generate context-aware answers.

Citations – provides references for transparency and trust.

Text-to-Speech – streams audio while the answer is being generated.

Streamlit UI – user-friendly web interface for interaction.

📂 Project Structure
.
├── app.py              # Streamlit app (UI + voice/text interaction)
├── rag_pipeline.py     # RAG pipeline (retrieval + prompt building + LLM answering)
├── retrieval.py        # Document retrieval logic (embeddings + similarity search)
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── data/               # Knowledge base (legal documents, FAQs, etc.)


# 📚 Tech Stack

Python

PyTorch – for LLM inference

SentenceTransformers – for embeddings & retrieval

Transformers (Hugging Face) – for language model pipeline

Streamlit – for the web UI

gTTS / pyttsx3 (TTS) – for text-to-speech


# 🛠️ Future Enhancements

Add multi-turn conversation support.

Support multilingual queries.

Deploy on cloud (AWS/GCP) with GPU acceleration.

Integrate speech-to-text (STT) for hands-free interaction.
