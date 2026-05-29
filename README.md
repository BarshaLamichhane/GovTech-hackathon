# GovTech AI Knowledge Bot

A Streamlit-based retrieval augmented generation (RAG) assistant for exploring GovTech Sandbox reports, local LLM wiki pages, and AI risk-management references.

The app builds a local FAISS vector index from available Markdown/PDF sources, then uses an LLM to answer questions, assess AI project readiness, and summarize recurring risks for public-sector AI pilots.

## Features

- Chat with GovTech Sandbox reports and wiki pages.
- Upload one or more PDF reports and include them in the knowledge base.
- Generate structured wiki entries from uploaded PDFs into a cloned `govtech-sandbox-hub` repository.
- Analyze planned AI projects with a readiness advisor.
- Create a risk dashboard from available wiki pages.
- Load optional MIT AI Risk Repository and NIST AI RMF reference files from `external/risk_sources/`.

## Project Structure

```text
.
├── app-govtech.py                 # Main GovTech Streamlit app
├── app.py                         # Older/simple PDF chat app
├── fileingestorGovTech.py         # GovTech ingestion, RAG, wiki, and risk logic
├── loadllm.py                     # LLM provider configuration
├── requirements.txt               # Python dependencies
├── wiki_pages/                    # Local Markdown knowledge base
├── external/
│   ├── govtech-sandbox-hub/       # Optional cloned Sandbox Hub repository
│   └── risk_sources/              # Optional MIT/NIST risk sources
└── vectorstore/                   # Generated FAISS index
```

## Requirements

- Python 3.10 or 3.11 recommended.
- A Mistral API key for the default LLM configuration.
- Optional: a Gemini API key if you want to use the Gemini loader in `loadllm.py`.

If you run into FAISS installation issues, use Python 3.11 or lower.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Or with conda:

```bash
conda create -n govtech-ai python=3.11 -y
conda activate govtech-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Clone the GovTech Sandbox Hub repository into `external/` so the app can load the full Sandbox wiki and reports:

```bash
mkdir -p external
git clone https://github.com/bowen-0/govtech-sandbox-hub.git external/govtech-sandbox-hub
```

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key
MISTRAL_API_MODEL_NAME=mistral-large-latest

# Optional, only needed if using Loadllm.load_gemini_llm()
GEMINI_API_KEY=your_gemini_api_key
GEMINI_API_MODEL_NAME=gemini-1.5-pro
```

## Run the App

Start the main GovTech app:

```bash
streamlit run app-govtech.py
```

Then open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

The sidebar lets you choose a mode and optionally upload PDF reports.

## App Modes

### Chat with reports

Ask questions about:

- Existing Markdown files under `wiki_pages/`
- The cloned Sandbox Hub wiki under `external/govtech-sandbox-hub/wiki/`
- Uploaded PDF reports
- Optional risk sources under `external/risk_sources/`

### AI Project Readiness Advisor

Describe a planned AI project. The app retrieves relevant context and returns:

- Similar Sandbox projects
- Legal, data protection, technical, and organizational risks
- Recommended next steps
- A readiness checklist

### Risk Dashboard

Summarizes recurring risks from available wiki pages, including legal, technical, organizational, and governance themes.

## Data Sources

The app automatically loads content from these locations when they exist:

```text
wiki_pages/
external/govtech-sandbox-hub/wiki/
external/risk_sources/
uploaded PDFs from the Streamlit sidebar
```

Supported external risk-source file types:

- `.md`
- `.markdown`
- `.txt`
- `.csv`
- `.json`
- `.pdf`

For more detail, see `external/risk_sources/README.md`.

## Updating the Sandbox Hub Wiki

If `external/govtech-sandbox-hub/` exists and you upload PDFs, the app can generate structured Markdown pages into:

```text
external/govtech-sandbox-hub/wiki/
```

Use the in-app button:

```text
Update cloned LLM Wiki from uploaded PDFs
```

Review generated files before committing or pushing them.

## Generated Files

The FAISS vector database is generated at:

```text
vectorstore/db_faiss
```

This directory can be deleted and rebuilt by rerunning the app.

Large local artifacts such as model files, vector stores, virtual environments, logs, and secrets are ignored by `.gitignore`.

## Optional Local LLM Download

`model_download.py` contains an older helper for downloading a Llama 2 GGUF/GGML model into `models/`. The current default app path uses Mistral through `loadllm.py`, so this is only needed if you switch the app back to a local `llama-cpp-python` model.

## Deployment Notes

`Service.md` contains an example `systemd` setup for running Streamlit as a Linux service. Update paths, user names, and environment names before using it on a server.

## Troubleshooting

- `No content found`: add Markdown files to `wiki_pages/`, clone `govtech-sandbox-hub` under `external/`, upload PDFs, or add files to `external/risk_sources/`.
- FAISS install errors: recreate the environment with Python 3.11 or 3.10.
- LLM authentication errors: check that `.env` contains `MISTRAL_API_KEY` and `MISTRAL_API_MODEL_NAME`.
- Slow first run: embedding models and dependencies may download/cache on first use.
