# Model-Powered RAG Learning Lab

This separate lab explains the local retrieval stack used by `Azure-RAG/backend/rag_service.py`:

```text
RecursiveCharacterTextSplitter
  -> sentence-transformers/all-MiniLM-L6-v2
  -> 384-dimensional chunk embeddings
  -> LangChain FAISS wrapper
  -> index.faiss + index.pkl
  -> query embedding
  -> FAISS nearest-neighbor search
  -> retrieved text context
```

It does not call an LLM. The goal is to understand the embedding encoder and retrieval database before generation.

## Run

Backend:

```bash
cd learning/model-powered/backend
python3 -m pip install -r requirements.txt
uvicorn main:app --reload --port 8002
```

Frontend and option selector:

```bash
cd learning
python3 -m http.server 3000
```

Open:

- Learning options: [http://127.0.0.1:3000/options/](http://127.0.0.1:3000/options/)
- Model-powered lab: [http://127.0.0.1:3000/model-powered/frontend/](http://127.0.0.1:3000/model-powered/frontend/)

The first run downloads `sentence-transformers/all-MiniLM-L6-v2`.

Both learning labs support default example text, direct copy/paste editing, and extraction from one or more PDFs. Multiple PDF texts are combined with visible document-name headers before chunking and indexing. Scanned image-only PDFs require OCR, which is intentionally outside this learning project.

## Vector Database Lifecycle

Before indexing, the UI inspects the saved database and recommends one of three operations:

- **Use existing:** loads the saved database without recomputing chunks or embeddings.
- **Recreate:** deletes the saved database and indexes every supplied document again. Use this after changing the embedding model, chunk size, overlap, or splitter.
- **Update existing:** computes SHA-256 hashes, skips exact duplicates, replaces changed documents, and adds new documents.

Uploaded PDFs are hashed from their original file bytes. Pasted/default text is hashed from its text content. The persisted `vectorstore_demo/registry.json` records filename, SHA-256 hash, chunk count, chunk IDs, UTC indexing timestamp, and indexing configuration.

The UI recommends `recreate` when configuration changes, `update` when new or changed documents are detected, and `use existing` when all supplied hashes are already registered.

An index created by an older version of this lab may contain `index.faiss` and `index.pkl` but no `registry.json`. Recreate that legacy index once so later updates can safely compare configuration and document hashes.

## What the Encoder Does

The same sentence-transformer independently encodes both chunks and questions:

```text
text
 -> tokenizer
 -> token IDs + attention mask
 -> MiniLM transformer
 -> contextual token representations
 -> sentence-transformer pooling
 -> one 384-dimensional semantic vector
```

Unlike the scratch lab, vector dimensions do not map to named vocabulary words. Meaning is distributed across all 384 values.

This is called a **bi-encoder** retrieval pattern because chunks and questions are encoded separately, then compared as vectors.

## What FAISS Stores

LangChain's local FAISS database uses two files:

### `index.faiss`

- Binary FAISS index
- Contains numeric vectors and nearest-neighbor index structure
- Does not contain readable original chunk text or metadata

### `index.pkl`

- LangChain document store
- Original chunk text
- Chunk metadata
- Mapping from integer FAISS positions to LangChain document IDs

Only load `index.pkl` from a trusted source because Python pickle files can execute code during deserialization.

### `registry.json`

- Document-level traceability outside FAISS
- Duplicate detection through SHA-256 hashes
- Chunk IDs needed to replace a changed document
- Configuration checks that prevent mixing incompatible vectors or chunks

## How Retrieval Works

The default LangChain FAISS index used here is `IndexFlatL2`.

```text
squared L2 distance = Σ(query_dimension - chunk_dimension)²
```

Lower distance means the query and chunk vectors are closer. `IndexFlatL2` performs exact search over every stored vector. It is simple and accurate for small local indexes, but larger production systems may use approximate indexes.

## Relationship to Azure-RAG

The model name, embedding wrapper, text splitter family, FAISS wrapper, and local persistence pattern match the current Azure-RAG implementation. The lab uses smaller default chunk sizes only so it produces multiple visible chunks from a short teaching document.

## Recursive Chunking

The frontend visualizes `RecursiveCharacterTextSplitter`, which tries separators in this order:

```text
paragraph break: "\n\n"
line break:      "\n"
space:           " "
characters:      ""
```

It prefers meaningful boundaries such as paragraphs and words. If a section remains too large, it recursively tries the next smaller separator.

For every generated chunk, the lab displays:

- Original-document start and end character positions
- Actual character count
- Repeated overlap text from the previous chunk
- Chunk text passed to the tokenizer and embedding encoder

## Nearest Does Not Always Mean Relevant

FAISS always returns the closest stored vectors, even for an unrelated question. Without another check, an irrelevant nearest neighbor could be sent to an LLM and encourage hallucination.

The model-powered lab adds a configurable maximum squared-L2 distance:

```text
FAISS nearest neighbors
  -> maximum L2-distance threshold
  -> accept relevant chunks or reject all chunks
```

Lower squared-L2 distance means closer vectors. The teaching default is `1.2` for this model and sample data. If every nearest neighbor is farther away, the UI:

- Shows the rejected candidates
- Sends no context to the LLM
- Returns the safe response: `I don't know based on the indexed document.`

There is no universal correct threshold. Production systems should calibrate it against a labeled evaluation dataset and may combine it with reranking, citations, and answer verification.
