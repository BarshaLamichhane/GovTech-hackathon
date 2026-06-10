# Learn RAG Step by Step

This folder contains two interactive projects that make the main stages of retrieval-augmented generation visible and understandable:

- **Scratch lab:** implements chunking, embeddings, vector storage, and cosine similarity directly without LangChain, FAISS, or an embedding model.
- **Model-powered lab:** uses the same Hugging Face MiniLM embeddings, LangChain splitter, and FAISS pattern as the Azure-RAG project.

Neither lab calls an LLM. They stop after retrieval so you can inspect exactly what context would be sent to one.

## What You Can Explore

1. **Chunking** splits a document into smaller overlapping sections.
2. **Embedding** converts each chunk into a numeric vector.
3. **Vector storage** keeps vectors together with their original text.
4. **Similarity search** finds chunks whose vectors are closest to the question vector.
5. **Retrieved context** shows the text that would be sent to an LLM.

The interface now shows both halves of RAG:

```text
INDEXING TIME
document -> chunk boundaries -> vocabulary -> chunk embeddings -> vector-store records

QUERY TIME
question -> question embedding -> cosine similarity -> top chunks -> LLM context
```

## Architecture

```text
learning/
├── scratch/
│   ├── backend/         # RAG implemented from scratch
│   └── frontend/
├── model-powered/
│   ├── backend/         # Hugging Face and FAISS learning API
│   └── frontend/
└── options/             # Page for choosing either lab
```

## Complete Running Guide

Run commands from the `GovTech-hackathon` repository root unless a command changes directories.

### Quick Start

For a complete setup, run these in three terminals:

```bash
# Terminal 1
cd learning/scratch/backend
python3 -m pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

```bash
# Terminal 2
cd learning/model-powered/backend
python3 -m pip install -r requirements.txt
uvicorn main:app --reload --port 8002
```

```bash
# Terminal 3
cd learning
python3 -m http.server 3000
```

Then open [http://127.0.0.1:3000/options/](http://127.0.0.1:3000/options/).

### Prerequisites

- Python 3.10 or newer
- Internet access during the first model-powered run so Hugging Face can download MiniLM
- Three terminal windows when running both labs together

Check Python:

```bash
python3 --version
```

### Optional: Create a Virtual Environment

Using one virtual environment for both learning backends:

```bash
cd learning
python3 -m venv .venv
source .venv/bin/activate
cd ..
```

On Windows PowerShell:

```powershell
cd learning
python -m venv .venv
.\.venv\Scripts\Activate.ps1
cd ..
```

Activate the environment again whenever opening a new backend terminal:

```bash
source learning/.venv/bin/activate
```

### Install Both Backends

Install the scratch backend:

```bash
python3 -m pip install -r learning/scratch/backend/requirements.txt
```

Install the model-powered backend:

```bash
python3 -m pip install -r learning/model-powered/backend/requirements.txt
```

The model-powered installation includes Hugging Face sentence transformers, LangChain, and FAISS.

### Terminal 1: Start the Scratch Backend

```bash
cd learning/scratch/backend
uvicorn main:app --reload --port 8001
```

Verify it:

```bash
curl http://127.0.0.1:8001/api/health
```

Expected response:

```json
{"status":"ok","indexed_chunks":0}
```

### Terminal 2: Start the Model-Powered Backend

```bash
cd learning/model-powered/backend
uvicorn main:app --reload --port 8002
```

The first startup downloads `sentence-transformers/all-MiniLM-L6-v2`. It can take a little longer than later startups.

Verify it:

```bash
curl http://127.0.0.1:8002/api/health
```

Expected response includes:

```json
{
  "status": "ok",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_dimensions": 384
}
```

### Terminal 3: Start the Shared Frontend Server

The frontend must be served from the `learning` directory so navigation between both labs works:

```bash
cd learning
python3 -m http.server 3000
```

Open these URLs:

- Learning options: [http://127.0.0.1:3000/options/](http://127.0.0.1:3000/options/)
- Scratch lab: [http://127.0.0.1:3000/scratch/frontend/](http://127.0.0.1:3000/scratch/frontend/)
- Model-powered lab: [http://127.0.0.1:3000/model-powered/frontend/](http://127.0.0.1:3000/model-powered/frontend/)

Each lab contains links for switching directly to the other lab.

### Choose Document Input

Both labs provide the same three document-input options:

1. **Default example:** click `Use default text` to restore the included teaching document.
2. **Copy and paste:** type, paste, or edit text directly in the document text area.
3. **Upload PDFs:** choose one or more PDFs and click `Extract selected PDFs`.

PDF extraction places the combined extracted text into the editable text area. Each source is separated with a header:

```text
=== DOCUMENT: example-one.pdf ===

Extracted text...

=== DOCUMENT: example-two.pdf ===

Extracted text...
```

The UI also displays page counts for each selected PDF. You can review or modify the combined text before clicking the indexing button.

The learning backends extract selectable text only. Image-only or scanned PDFs require OCR and display a message explaining that no selectable text was found.

### Choose a Model-Powered Index Operation

The model-powered lab persists its FAISS database between backend runs. Before running the pipeline, choose:

1. **Use existing** to load saved vectors without recomputing embeddings.
2. **Recreate** to rebuild the complete database.
3. **Update existing** to hash documents, skip duplicates, replace changed files, and add new files.

The page recommends recreating after chunking or embedding configuration changes. It recommends updating when it detects new or changed documents. The registry is stored at `learning/model-powered/backend/vectorstore_demo/registry.json` and shows filename, SHA-256 hash, chunk count, and indexing timestamp.

### Run Only One Lab

For the scratch lab, run:

```bash
cd learning/scratch/backend
uvicorn main:app --reload --port 8001
```

For the model-powered lab, run:

```bash
cd learning/model-powered/backend
uvicorn main:app --reload --port 8002
```

In either case, also run the shared frontend server:

```bash
cd learning
python3 -m http.server 3000
```

### Run Tests

Scratch tests:

```bash
cd learning/scratch/backend
pytest -q
```

Model-powered tests:

```bash
cd learning/model-powered/backend
pytest -q
```

The model-powered tests load MiniLM and therefore take longer.

### Stop the Servers

Press `Ctrl+C` inside each terminal running Uvicorn or the frontend HTTP server.

### Common Problems

#### Address already in use

Another process is already using the requested port. Stop the old process or choose another port.

The frontend JavaScript currently expects:

```text
Scratch API:       http://127.0.0.1:8001/api
Model-powered API: http://127.0.0.1:8002/api
```

If you change backend ports, update the corresponding frontend `app.js`.

#### Model-powered page cannot connect

Confirm the model-powered backend is running:

```bash
curl http://127.0.0.1:8002/api/health
```

#### Scratch page cannot connect

Confirm the scratch backend is running:

```bash
curl http://127.0.0.1:8001/api/health
```

#### Hugging Face download warning

The first model-powered run downloads MiniLM from Hugging Face. An unauthenticated-request warning is acceptable for this learning project, though downloads may be rate-limited.

#### Do not open HTML files directly

Use `python3 -m http.server 3000` instead of opening `index.html` with a `file://` URL. The shared HTTP server preserves navigation paths and lets the frontend call the APIs consistently.

## Read the Code in This Order

For the scratch implementation, open `scratch/backend/main.py` and follow these functions:

| Function or class | Concept |
| --- | --- |
| `chunk_text` | Splits a document using chunk size and overlap |
| `build_vocabulary` | Creates the dimensions used by all vectors |
| `embed` | Creates a normalized bag-of-words embedding |
| `cosine_similarity` | Measures similarity between two vectors |
| `SimpleVectorStore` | Stores chunks and vectors, then retrieves matches |
| `make_extractive_answer` | Stands in for an LLM so retrieval remains visible |

For the real-model implementation, open `model-powered/backend/main.py` and follow:

| Function or class | Concept |
| --- | --- |
| `ModelRagLab.index` | Recursive chunking, MiniLM embedding, and FAISS indexing |
| `ModelRagLab.recommendation` | Recommends use, recreate, or update from configuration and document hashes |
| `ModelRagLab.tokenize` | MiniLM tokenizer output |
| `inspect_saved_files` | Contents of `index.faiss`, `index.pkl`, and `registry.json` |
| `search` | Query embedding, FAISS search, and relevance filtering |

## Indexing Time Versus Query Time

**Indexing** happens before users ask questions:

1. Split documents into chunks.
2. Build embeddings for every chunk.
3. Save each chunk, embedding, ID, and metadata in the vector store.

**Querying** happens whenever a user asks something:

1. Embed the question using the same vector dimensions.
2. Compare the question vector with every chunk vector.
3. Retrieve the chunks with the highest similarity.
4. Send those chunks to an LLM as context.

The interface displays both pipelines, including chunk word ranges, overlap, raw word counts, normalized vectors, vector-store records, question-vector dimensions, and cosine-similarity calculations.

## Understanding the Embeddings

Suppose the vocabulary is:

```text
["battery", "solar", "stores"]
```

The text `"solar battery battery"` becomes a word-count vector:

```text
[2, 1, 0]
```

The demo normalizes this vector so longer chunks do not automatically appear more relevant.

Real embedding models create dense semantic vectors. They can recognize that `"car"` and `"automobile"` are related even when the words differ. This demo uses word counts, so matching words matter directly.

It also treats forms such as `"store"` and `"stores"` as different words. Try these variants in the interface to see why semantic embedding models are valuable.

## How This Maps to Production RAG

| Learning project | Production equivalent |
| --- | --- |
| `chunk_text` | LangChain text splitter or Azure AI Search skillset |
| `embed` | Hugging Face, Gemini, OpenAI, or Azure OpenAI embeddings |
| `SimpleVectorStore` | FAISS, Azure AI Search, Pinecone, or another vector database |
| `cosine_similarity` | Vector database nearest-neighbor search |
| Extractive answer | Gemini, Azure OpenAI, or another LLM |

## Important Simplifications

- The vector store exists only in memory and resets when the backend restarts.
- Embeddings understand matching words, not meaning.
- Only one document is indexed at a time.
- The final answer does not call an LLM.

These limitations are useful here: each RAG step remains small enough to understand completely.

## Why Irrelevant Questions Still Have Nearest Neighbors

A vector search always ranks the stored chunks. Even when a question is completely unrelated, one chunk will still be mathematically closest.

The scratch lab therefore uses two stages:

```text
nearest-neighbor search
  -> minimum cosine-similarity threshold
  -> accept relevant chunks or reject all chunks
```

The default minimum similarity is `0.2`. If no chunk reaches it, the UI still shows the rejected nearest neighbors for inspection, but sends no context to an LLM and returns:

```text
I don't know based on the indexed document.
```

Thresholds are application decisions. A production system should calibrate them using representative questions and documents.
