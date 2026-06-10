import re
from collections import Counter
from io import BytesIO
from typing import List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="RAG Learning Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class LearnRequest(BaseModel):
    text: str = Field(min_length=1)
    chunk_size: int = Field(default=35, ge=5, le=200)
    chunk_overlap: int = Field(default=5, ge=0, le=100)


class SearchRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    minimum_similarity: float = Field(default=0.2, ge=0, le=1)


class SimpleVectorStore:
    """A tiny in-memory vector store for learning purposes."""

    def __init__(self):
        self.chunks: List[str] = []
        self.chunk_records: List[dict] = []
        self.vocabulary: List[str] = []
        self.vectors = np.empty((0, 0))

    def index(self, text: str, chunk_size: int, chunk_overlap: int):
        self.chunk_records = chunk_text_with_details(text, chunk_size, chunk_overlap)
        self.chunks = [record["text"] for record in self.chunk_records]
        self.vocabulary = build_vocabulary(self.chunks)
        self.vectors = np.array(
            [embed(chunk, self.vocabulary) for chunk in self.chunks],
            dtype=float,
        )

    def search(self, question: str, top_k: int):
        if not self.chunks:
            raise ValueError("Index a document before searching.")

        question_vector = embed(question, self.vocabulary)
        question_counts = word_counts(question, self.vocabulary)
        question_magnitude = vector_magnitude(question_counts)
        scored_chunks = []

        for index, (chunk, vector, record) in enumerate(
            zip(self.chunks, self.vectors, self.chunk_records)
        ):
            dot_product = float(np.dot(question_vector, vector))
            score = cosine_similarity(question_vector, vector)
            scored_chunks.append(
                {
                    "chunk_id": index,
                    "text": chunk,
                    "score": round(score, 4),
                    "vector": vector.round(3).tolist(),
                    "dot_product": round(dot_product, 4),
                    "question_vector_magnitude": round(float(np.linalg.norm(question_vector)), 4),
                    "chunk_vector_magnitude": round(float(np.linalg.norm(vector)), 4),
                    "start_word": record["start_word"],
                    "end_word": record["end_word"],
                }
            )

        scored_chunks.sort(key=lambda item: item["score"], reverse=True)
        return {
            "question_tokens": tokenize(question),
            "question_word_counts": question_counts.tolist(),
            "question_raw_magnitude": round(question_magnitude, 4),
            "question_vector": question_vector,
            "retrieved": scored_chunks[:top_k],
            "all_scores": scored_chunks,
        }


store = SimpleVectorStore()


def extract_pdf_text(contents: bytes) -> dict:
    from pypdf import PdfReader

    try:
        reader = PdfReader(BytesIO(contents))
    except Exception as exc:
        raise ValueError("Could not read this PDF file.") from exc

    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": page_number, "text": text})

    if not pages:
        raise ValueError(
            "No selectable text was found. Scanned PDFs require OCR, which this learning app does not perform."
        )

    return {
        "page_count": len(reader.pages),
        "pages_with_text": len(pages),
        "text": "\n\n".join(page["text"] for page in pages),
        "page_previews": [
            {"page": page["page"], "preview": page["text"][:180]}
            for page in pages
        ],
    }


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def chunk_text_with_details(text: str, chunk_size: int, chunk_overlap: int) -> List[dict]:
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    words = text.split()
    chunks = []
    step = chunk_size - chunk_overlap

    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_size]
        if chunk:
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "text": " ".join(chunk),
                    "words": chunk,
                    "start_word": start,
                    "end_word": start + len(chunk) - 1,
                    "overlap_from_previous": chunk[:chunk_overlap] if start > 0 else [],
                }
            )
        if start + chunk_size >= len(words):
            break

    return chunks


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    return [
        record["text"]
        for record in chunk_text_with_details(text, chunk_size, chunk_overlap)
    ]


def build_vocabulary(chunks: List[str]) -> List[str]:
    return sorted({token for chunk in chunks for token in tokenize(chunk)})


def word_counts(text: str, vocabulary: List[str]) -> np.ndarray:
    counts = Counter(tokenize(text))
    return np.array([counts[word] for word in vocabulary], dtype=float)


def vector_magnitude(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def embed(text: str, vocabulary: List[str]) -> np.ndarray:
    """Create a normalized bag-of-words vector."""
    vector = word_counts(text, vocabulary)
    magnitude = vector_magnitude(vector)
    return vector / magnitude if magnitude else vector


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(np.dot(first, second) / denominator) if denominator else 0.0


def make_extractive_answer(question: str, retrieved_chunks: List[dict]) -> str:
    if not retrieved_chunks or retrieved_chunks[0]["score"] == 0:
        return "The indexed document does not contain enough matching words to answer."

    best_chunk = retrieved_chunks[0]["text"]
    sentences = re.split(r"(?<=[.!?])\s+", best_chunk)
    question_words = set(tokenize(question))
    best_sentence = max(
        sentences,
        key=lambda sentence: len(question_words.intersection(tokenize(sentence))),
    )
    return best_sentence


@app.get("/api/health")
def health():
    return {"status": "ok", "indexed_chunks": len(store.chunks)}


@app.post("/api/extract-pdfs")
async def extract_pdfs(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one PDF file.")

    extracted_files = []
    combined_sections = []
    for file in files:
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename}: upload PDF files only.")
        try:
            result = extract_pdf_text(await file.read())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"{file.filename}: {exc}") from exc
        extracted_files.append({"filename": file.filename, **result})
        combined_sections.append(f"=== DOCUMENT: {file.filename} ===\n\n{result['text']}")

    return {
        "file_count": len(extracted_files),
        "total_pages": sum(item["page_count"] for item in extracted_files),
        "total_pages_with_text": sum(item["pages_with_text"] for item in extracted_files),
        "combined_text": "\n\n".join(combined_sections),
        "files": extracted_files,
    }


@app.post("/api/index")
def index_document(request: LearnRequest):
    try:
        store.index(request.text, request.chunk_size, request.chunk_overlap)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "explanation": {
            "chunking": "The document is split into overlapping groups of words.",
            "embedding": "Each chunk becomes a normalized word-count vector.",
            "vector_store": "The vectors and their original chunks are stored together in memory.",
        },
        "chunk_size": request.chunk_size,
        "chunk_overlap": request.chunk_overlap,
        "step_size": request.chunk_size - request.chunk_overlap,
        "document_words": request.text.split(),
        "document_word_count": len(request.text.split()),
        "vocabulary": store.vocabulary,
        "vocabulary_size": len(store.vocabulary),
        "indexing_steps": [
            {
                "step": 1,
                "name": "Tokenize and chunk",
                "result": f"Created {len(store.chunks)} chunks from {len(request.text.split())} words.",
            },
            {
                "step": 2,
                "name": "Build shared vocabulary",
                "result": f"Created {len(store.vocabulary)} vector dimensions.",
            },
            {
                "step": 3,
                "name": "Embed every chunk",
                "result": f"Created {len(store.vectors)} normalized vectors.",
            },
            {
                "step": 4,
                "name": "Write vector-store records",
                "result": f"Stored {len(store.chunks)} records in memory.",
            },
        ],
        "chunks": [
            {
                "chunk_id": index,
                "text": chunk,
                "words": record["words"],
                "start_word": record["start_word"],
                "end_word": record["end_word"],
                "overlap_from_previous": record["overlap_from_previous"],
                "word_counts": word_counts(chunk, store.vocabulary).tolist(),
                "raw_vector_magnitude": round(
                    vector_magnitude(word_counts(chunk, store.vocabulary)),
                    4,
                ),
                "vector": vector.round(3).tolist(),
            }
            for index, (chunk, vector, record) in enumerate(
                zip(store.chunks, store.vectors, store.chunk_records)
            )
        ],
        "vector_store_records": [
            {
                "record_id": f"chunk-{index}",
                "metadata": {
                    "chunk_id": index,
                    "start_word": record["start_word"],
                    "end_word": record["end_word"],
                    "vector_dimensions": len(vector),
                },
                "text": chunk,
                "vector": vector.round(3).tolist(),
            }
            for index, (chunk, vector, record) in enumerate(
                zip(store.chunks, store.vectors, store.chunk_records)
            )
        ],
    }


@app.post("/api/search")
def search(request: SearchRequest):
    try:
        search_result = store.search(request.question, request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    nearest_neighbors = search_result["retrieved"]
    retrieved = [
        item for item in nearest_neighbors
        if item["score"] >= request.minimum_similarity
    ]
    all_scores = search_result["all_scores"]
    has_relevant_context = bool(retrieved)
    return {
        "question": request.question,
        "question_tokens": search_result["question_tokens"],
        "question_word_counts": search_result["question_word_counts"],
        "question_raw_magnitude": search_result["question_raw_magnitude"],
        "question_vector": search_result["question_vector"].round(3).tolist(),
        "vocabulary": store.vocabulary,
        "question_embedding_steps": [
            "Tokenize the question.",
            "Count each vocabulary word in the question.",
            "Create a vector in the same dimension order as stored chunks.",
            "Normalize the vector so cosine similarity compares direction.",
        ],
        "minimum_similarity": request.minimum_similarity,
        "nearest_neighbors": nearest_neighbors,
        "retrieved_chunks": retrieved,
        "has_relevant_context": has_relevant_context,
        "retrieval_decision": (
            f"Accepted {len(retrieved)} chunk(s) with cosine similarity at or above "
            f"{request.minimum_similarity}."
            if has_relevant_context
            else (
                f"Rejected all nearest neighbors because none reached the minimum cosine "
                f"similarity of {request.minimum_similarity}."
            )
        ),
        "all_similarity_scores": all_scores,
        "context_sent_to_llm": "\n\n".join(item["text"] for item in retrieved),
        "simple_extractive_answer": (
            make_extractive_answer(request.question, retrieved)
            if has_relevant_context
            else "I don't know based on the indexed document."
        ),
        "explanation": (
            "The question is embedded with the same vocabulary. Cosine similarity compares "
            "its direction with every stored chunk vector. The highest scores are retrieved."
        ),
    }
