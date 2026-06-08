# Spider_MLtask
Project Overview

This repository contains solutions for both the Base Task and Applied ML Domain Task.

---

Task 1: Fashion-MNIST Classification

OBJECTIVE
Develop a neural network using PyTorch to classify Fashion-MNIST images into their corresponding clothing categories.

TECHNOLOGIES USED
- Python
- PyTorch
- Pandas
- NumPy
- Matplotlib

A separate README.md file included for this task in the respective folder

---

Task 2: Research Paper RAG Assistant

OBJECTIVE
Build a Retrieval-Augmented Generation (RAG) system capable of answering questions using information extracted from multiple NLP and LLM research papers.

RESOURCES
- Attention Is All You Need
- BERT
- GPT
- RAG
- Sentence-BERT
- LoRA
- Llama 2

RAG PIPELINE
1. Load PDF research papers using PyPDFLoader.
2. Split documents into chunks using RecursiveCharacterTextSplitter.
3. Generate embeddings using Sentence Transformers.
4. Store embeddings in ChromaDB.
5. Retrieve relevant chunks using semantic similarity search.
6. Generate responses using Gemini.
7. Display answers along with source references.
   
FEATURES
- Multi-document retrieval
- Semantic search
- Source-aware responses
- Hallucination reduction through retrieval
- Streamlit-based user interface

TECHNOLOGIES USED
- Python
- LangChain
- ChromaDB
- Sentence Transformers
- Google Gemini
- Streamlit

