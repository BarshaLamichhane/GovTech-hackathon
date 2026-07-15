# GovTech Hackathon Workspace

This repository is now split into two main folders:

- `gov-tech-RAG-wiki/` — contains the existing GovTech RAG/Wiki prototype and legacy app logic.
- `Azure-RAG/` — contains the new React frontend and backend for a PDF-based Azure RAG chatbot. An Azure-ready RAG document QA system with FastAPI, Hugging Face embeddings, FAISS, multi-PDF ingestion, vector index management, and grounded LLM-style retrieval.
- `learning/` — contains a minimal interactive project for understanding chunking, embeddings, vector stores, and retrieval.

## How to use

- `gov-tech-RAG-wiki/` includes the older project files, wiki ingestion logic, and legacy Streamlit apps.
- `Azure-RAG/backend/` provides a FastAPI backend for uploading PDFs and asking retrieval-based questions.
- `Azure-RAG/frontend/` provides a React UI for the new chatbot.

For details on the new prototype, open `Azure-RAG/backend/README.md` and `Azure-RAG/frontend/README.md`.
