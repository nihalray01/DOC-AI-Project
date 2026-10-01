# DocuMind AI

DocuMind AI is an intelligent PDF question-answering system
built using Retrieval-Augmented Generation (RAG).

Users can upload one or more PDF documents and ask questions
based on their content. The system retrieves the most relevant
document sections using semantic search and uses Gemini AI to
generate grounded answers with source references.

## Features

- Multiple PDF upload
- Page-wise PDF text extraction
- Overlapping text chunking
- Sentence Transformer embeddings
- FAISS vector similarity search
- Gemini AI integration
- Source file and page references
- Chat-style Streamlit interface
- Retrieval-only fallback mode
- English and Hindi question support

## Technology Stack

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- FAISS
- Gemini API
- NumPy

## Project Architecture

PDF Upload  
→ Text Extraction  
→ Text Chunking  
→ Embedding Generation  
→ FAISS Vector Index  
→ User Question  
→ Semantic Retrieval  
→ Gemini Answer Generation  
→ Answer with Sources

## Installation

Create and activate a virtual environment:

```powershell
py -m venv venv
venv\Scripts\activate