import hashlib
import json
import pickle
import shutil
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import List, Literal, Optional

import faiss
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "vectorstore_demo"
REGISTRY_PATH = DB_PATH / "registry.json"

app = FastAPI(title="Model-Powered RAG Learning Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SourceDocument(BaseModel):
    filename: str = Field(min_length=1)
    text: str = Field(min_length=1)
    file_hash: Optional[str] = None


class IndexRequest(BaseModel):
    operation: Literal["use_existing", "recreate", "update"] = "recreate"
    documents: List[SourceDocument] = Field(default_factory=list)
    text: Optional[str] = None
    chunk_size: int = Field(default=220, ge=50, le=1000)
    chunk_overlap: int = Field(default=40, ge=0, le=500)


class RecommendationRequest(BaseModel):
    documents: List[SourceDocument] = Field(default_factory=list)
    text: Optional[str] = None
    chunk_size: int = Field(default=220, ge=50, le=1000)
    chunk_overlap: int = Field(default=40, ge=0, le=500)


class SearchRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    maximum_l2_distance: float = Field(default=1.2, ge=0, le=4)


class ModelRagLab:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            encode_kwargs={"normalize_embeddings": False},
        )
        self.db = None
        self.chunks: List[Document] = []
        self.vectors = np.empty((0, 384), dtype=np.float32)

    @property
    def encoder(self):
        return self.embeddings._client

    def tokenize(self, text: str):
        encoded = self.encoder.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            return_attention_mask=True,
        )
        return {
            "tokens": self.encoder.tokenizer.convert_ids_to_tokens(encoded["input_ids"]),
            "token_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def configuration(self, chunk_size: int, chunk_overlap: int):
        return {
            "embedding_model": MODEL_NAME,
            "embedding_dimensions": self.encoder.get_embedding_dimension(),
            "splitter": "RecursiveCharacterTextSplitter",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }

    def load_registry(self):
        if not REGISTRY_PATH.exists():
            return {"configuration": None, "documents": []}
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        registry.setdefault("configuration", None)
        registry.setdefault("documents", [])
        return registry

    def save_registry(self, registry):
        DB_PATH.mkdir(parents=True, exist_ok=True)
        REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    def has_saved_index(self):
        return (DB_PATH / "index.faiss").exists() and (DB_PATH / "index.pkl").exists()

    def configuration_changes(self, saved_configuration, requested_configuration):
        if not saved_configuration:
            return ["saved configuration is unavailable"]
        return [
            key
            for key, value in requested_configuration.items()
            if saved_configuration.get(key) != value
        ]

    def load_existing(self):
        if not self.has_saved_index():
            raise ValueError("No existing vector database was found.")
        self.db = FAISS.load_local(
            str(DB_PATH),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.refresh_from_db()

    def refresh_from_db(self):
        if self.db is None:
            self.chunks = []
            self.vectors = np.empty((0, self.encoder.get_embedding_dimension()), dtype=np.float32)
            return
        self.chunks = [
            self.db.docstore.search(self.db.index_to_docstore_id[position])
            for position in sorted(self.db.index_to_docstore_id)
        ]
        self.vectors = np.array(
            [self.db.index.reconstruct(position) for position in range(self.db.index.ntotal)],
            dtype=np.float32,
        )

    def normalized_documents(self, documents: List[SourceDocument], text: Optional[str]):
        if documents:
            return [
                {
                    "filename": document.filename,
                    "text": document.text,
                    "file_hash": document.file_hash or hashlib.sha256(document.text.encode()).hexdigest(),
                }
                for document in documents
            ]
        if text and text.strip():
            return [{
                "filename": "pasted-or-default-text.txt",
                "text": text,
                "file_hash": hashlib.sha256(text.encode()).hexdigest(),
            }]
        return []

    def split_documents(self, documents, chunk_size: int, chunk_overlap: int):
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size.")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
        chunks = []
        chunk_ids = []
        per_document = []
        for document in documents:
            source = Document(
                page_content=document["text"],
                metadata={"source": document["filename"], "file_hash": document["file_hash"]},
            )
            document_chunks = splitter.split_documents([source])
            ids = []
            for index, chunk in enumerate(document_chunks):
                chunk_id = hashlib.sha256(
                    f"{document['file_hash']}:{index}:{chunk.page_content}".encode()
                ).hexdigest()
                chunk.metadata["chunk_id"] = index
                chunk.metadata["document_chunk_id"] = chunk_id
                chunks.append(chunk)
                chunk_ids.append(chunk_id)
                ids.append(chunk_id)
            per_document.append({**document, "chunk_ids": ids, "chunk_count": len(ids)})
        return chunks, chunk_ids, per_document

    def recommendation(self, documents, chunk_size: int, chunk_overlap: int):
        registry = self.load_registry()
        configuration = self.configuration(chunk_size, chunk_overlap)
        if not self.has_saved_index():
            return {"recommended_operation": "recreate", "reason": "No existing vector database was found."}
        if not registry["configuration"] or not registry["documents"]:
            return {
                "recommended_operation": "recreate",
                "reason": (
                    "The existing vector database was created before registry tracking was added. "
                    "Recreate it once to record its documents and configuration."
                ),
            }
        changed_fields = self.configuration_changes(registry["configuration"], configuration)
        if changed_fields:
            return {
                "recommended_operation": "recreate",
                "reason": (
                    "Index configuration changed for: "
                    f"{', '.join(changed_fields)}. Existing vectors are not comparable or consistently chunked."
                ),
                "changed_configuration_fields": changed_fields,
            }
        known_hashes = {item["file_hash"] for item in registry["documents"]}
        known_by_name = {item["filename"]: item for item in registry["documents"]}
        new_documents = [
            item["filename"] for item in documents
            if item["filename"] not in known_by_name and item["file_hash"] not in known_hashes
        ]
        changed_documents = [
            item["filename"] for item in documents
            if item["filename"] in known_by_name and item["file_hash"] != known_by_name[item["filename"]]["file_hash"]
        ]
        if new_documents or changed_documents:
            return {
                "recommended_operation": "update",
                "reason": "New or changed documents were detected. Update adds them without rebuilding unchanged documents.",
                "new_documents": new_documents,
                "changed_documents": changed_documents,
            }
        return {
            "recommended_operation": "use_existing",
            "reason": "The configuration matches and every supplied document hash already exists.",
        }

    def index(self, operation: str, documents, chunk_size: int, chunk_overlap: int):
        configuration = self.configuration(chunk_size, chunk_overlap)
        now = datetime.now(timezone.utc).isoformat()
        duplicate_filenames = []
        unique_documents = []
        seen_hashes = set()
        for document in documents:
            if document["file_hash"] in seen_hashes:
                duplicate_filenames.append(document["filename"])
            else:
                seen_hashes.add(document["file_hash"])
                unique_documents.append(document)
        documents = unique_documents

        if operation == "use_existing":
            self.load_existing()
            return {"message": "Loaded the existing vector database without recomputing embeddings.", "skipped_duplicates": []}

        if not documents:
            raise ValueError("Provide text or PDF documents before creating or updating the vector database.")

        if operation == "recreate":
            if DB_PATH.exists():
                shutil.rmtree(DB_PATH)
            chunks, ids, document_records = self.split_documents(documents, chunk_size, chunk_overlap)
            vectors = self.embeddings.embed_documents([chunk.page_content for chunk in chunks])
            self.db = FAISS.from_embeddings(
                [(chunk.page_content, vector) for chunk, vector in zip(chunks, vectors)],
                self.embeddings,
                metadatas=[chunk.metadata for chunk in chunks],
                ids=ids,
            )
            registry_documents = [
                {
                    "filename": item["filename"],
                    "file_hash": item["file_hash"],
                    "chunk_count": item["chunk_count"],
                    "chunk_ids": item["chunk_ids"],
                    "indexed_at": now,
                }
                for item in document_records
            ]
            message = f"Recreated the vector database with {len(registry_documents)} document(s)."
            skipped = duplicate_filenames
        else:
            if not self.has_saved_index():
                raise ValueError("No existing vector database exists. Choose recreate instead.")
            registry = self.load_registry()
            if not registry["configuration"] or not registry["documents"]:
                raise ValueError(
                    "This existing vector database has no document registry or saved configuration. "
                    "Choose Recreate once before using Update existing."
                )
            changed_fields = self.configuration_changes(registry["configuration"], configuration)
            if changed_fields:
                raise ValueError(
                    "Index configuration changed for "
                    f"{', '.join(changed_fields)}. Recreate the vector database before indexing."
                )
            self.load_existing()
            known_hashes = {item["file_hash"] for item in registry["documents"]}
            known_by_name = {item["filename"]: item for item in registry["documents"]}
            skipped = duplicate_filenames + [
                item["filename"] for item in documents if item["file_hash"] in known_hashes
            ]
            changed_names = {
                item["filename"] for item in documents
                if item["filename"] in known_by_name and item["file_hash"] != known_by_name[item["filename"]]["file_hash"]
            }
            delete_ids = [
                chunk_id
                for name in changed_names
                for chunk_id in known_by_name[name]["chunk_ids"]
            ]
            if delete_ids:
                self.db.delete(delete_ids)
            to_add = [
                item for item in documents
                if item["file_hash"] not in known_hashes
            ]
            if to_add:
                chunks, ids, document_records = self.split_documents(to_add, chunk_size, chunk_overlap)
                vectors = self.embeddings.embed_documents([chunk.page_content for chunk in chunks])
                self.db.add_embeddings(
                    [(chunk.page_content, vector) for chunk, vector in zip(chunks, vectors)],
                    metadatas=[chunk.metadata for chunk in chunks],
                    ids=ids,
                )
            else:
                document_records = []
            registry_documents = [
                item for item in registry["documents"]
                if item["filename"] not in changed_names
            ] + [
                {
                    "filename": item["filename"],
                    "file_hash": item["file_hash"],
                    "chunk_count": item["chunk_count"],
                    "chunk_ids": item["chunk_ids"],
                    "indexed_at": now,
                }
                for item in document_records
            ]
            message = f"Updated the vector database with {len(document_records)} new or changed document(s)."

        DB_PATH.mkdir(parents=True, exist_ok=True)
        self.db.save_local(str(DB_PATH))
        self.save_registry({"configuration": configuration, "documents": registry_documents, "updated_at": now})
        self.refresh_from_db()
        return {"message": message, "skipped_duplicates": skipped}

    def ensure_index(self):
        if self.db is None:
            raise ValueError("Create the model-powered FAISS index before searching.")


lab = ModelRagLab()


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


def vector_details(vector: np.ndarray):
    return {
        "dimensions": int(vector.shape[0]),
        "first_16_values": vector[:16].round(5).tolist(),
        "minimum": round(float(vector.min()), 6),
        "maximum": round(float(vector.max()), 6),
        "mean": round(float(vector.mean()), 6),
        "magnitude": round(float(np.linalg.norm(vector)), 6),
        "non_zero_values": int(np.count_nonzero(vector)),
    }


def longest_overlap(previous: str, current: str) -> str:
    max_length = min(len(previous), len(current))
    for length in range(max_length, 0, -1):
        if previous[-length:] == current[:length]:
            return current[:length]
    return ""


def pca_coordinates(vectors: np.ndarray):
    if len(vectors) == 1:
        return [[0.0, 0.0]]
    centered = vectors - vectors.mean(axis=0)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    dimensions = min(2, components.shape[0])
    projected = centered @ components[:dimensions].T
    if dimensions == 1:
        projected = np.column_stack([projected[:, 0], np.zeros(len(vectors))])
    return projected.round(5).tolist()


def inspect_saved_files():
    faiss_path = DB_PATH / "index.faiss"
    pickle_path = DB_PATH / "index.pkl"
    saved_index = faiss.read_index(str(faiss_path))
    with pickle_path.open("rb") as file:
        docstore, index_to_docstore_id = pickle.load(file)
    registry = lab.load_registry()
    vector_samples = [
        {
            "faiss_position": position,
            "first_12_values": saved_index.reconstruct(position)[:12].round(6).tolist(),
        }
        for position in range(min(saved_index.ntotal, 3))
    ]
    document_samples = [
        {
            "document_id": document_id,
            "faiss_position": next(
                (
                    position
                    for position, mapped_id in index_to_docstore_id.items()
                    if mapped_id == document_id
                ),
                None,
            ),
            "text": document.page_content,
            "metadata": document.metadata,
        }
        for document_id, document in docstore._dict.items()
    ]

    return {
        "index_faiss": {
            "path": str(faiss_path),
            "size_bytes": faiss_path.stat().st_size,
            "stores": "Binary FAISS index containing vectors and nearest-neighbor index structure.",
            "does_not_store": "Original chunk text or metadata.",
            "details": {
                "index_class": type(saved_index).__name__,
                "metric_type": "L2 distance" if saved_index.metric_type == faiss.METRIC_L2 else str(saved_index.metric_type),
                "dimensions_per_vector": saved_index.d,
                "total_vectors": saved_index.ntotal,
                "sample_vectors": vector_samples,
                "sample_note": "Only the first 12 values of the first 3 vectors are shown. The complete vectors remain in the binary file.",
            },
        },
        "index_pkl": {
            "path": str(pickle_path),
            "size_bytes": pickle_path.stat().st_size,
            "stores": "LangChain docstore with chunk text and metadata, plus FAISS-position-to-document-ID mapping.",
            "security_note": "Pickle files must only be loaded from trusted sources.",
            "document_ids": list(docstore._dict.keys()),
            "index_to_docstore_id": index_to_docstore_id,
            "details": {
                "document_count": len(docstore._dict),
                "mapping_count": len(index_to_docstore_id),
                "index_to_docstore_id": index_to_docstore_id,
                "documents": document_samples,
            },
        },
        "registry_json": {
            "path": str(REGISTRY_PATH),
            "size_bytes": REGISTRY_PATH.stat().st_size if REGISTRY_PATH.exists() else 0,
            "stores": "Document filenames, SHA-256 hashes, chunk IDs, chunk counts, timestamps, and indexing configuration.",
            "details": registry,
        },
    }


@app.get("/api/health")
def health():
    registry = lab.load_registry()
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "embedding_dimensions": lab.encoder.get_embedding_dimension(),
        "indexed_chunks": sum(item["chunk_count"] for item in registry["documents"]),
        "has_existing_vector_database": lab.has_saved_index(),
        "registered_documents": len(registry["documents"]),
    }


@app.get("/api/index/status")
def index_status():
    registry = lab.load_registry()
    return {
        "has_existing_vector_database": lab.has_saved_index(),
        "configuration": registry["configuration"],
        "documents": registry["documents"],
        "updated_at": registry.get("updated_at"),
    }


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
            contents = await file.read()
            result = extract_pdf_text(contents)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"{file.filename}: {exc}") from exc
        contents_hash = hashlib.sha256(contents).hexdigest()
        extracted_files.append({"filename": file.filename, "file_hash": contents_hash, **result})
        combined_sections.append(f"=== DOCUMENT: {file.filename} ===\n\n{result['text']}")

    return {
        "file_count": len(extracted_files),
        "total_pages": sum(item["page_count"] for item in extracted_files),
        "total_pages_with_text": sum(item["pages_with_text"] for item in extracted_files),
        "combined_text": "\n\n".join(combined_sections),
        "files": extracted_files,
    }


@app.post("/api/index/recommendation")
def index_recommendation(request: RecommendationRequest):
    documents = lab.normalized_documents(request.documents, request.text)
    return {
        **lab.recommendation(documents, request.chunk_size, request.chunk_overlap),
        "has_existing_vector_database": lab.has_saved_index(),
        "configuration": lab.configuration(request.chunk_size, request.chunk_overlap),
        "document_count": len(documents),
    }


@app.post("/api/index")
def index_document(request: IndexRequest):
    documents = lab.normalized_documents(request.documents, request.text)
    try:
        operation_result = lab.index(
            request.operation,
            documents,
            request.chunk_size,
            request.chunk_overlap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    registry = lab.load_registry()
    active_configuration = registry["configuration"] or lab.configuration(request.chunk_size, request.chunk_overlap)
    index = lab.db.index
    coordinates = pca_coordinates(lab.vectors)
    records = []
    for faiss_position, document_id in lab.db.index_to_docstore_id.items():
        document = lab.db.docstore.search(document_id)
        reconstructed = index.reconstruct(faiss_position)
        previous_text = records[-1]["text"] if records else ""
        overlap_text = longest_overlap(previous_text, document.page_content)
        start_index = document.metadata.get("start_index", 0)
        records.append(
            {
                "faiss_position": faiss_position,
                "document_id": document_id,
                "text": document.page_content,
                "metadata": document.metadata,
                "chunking": {
                    "start_character": start_index,
                    "end_character": start_index + len(document.page_content) - 1,
                    "character_count": len(document.page_content),
                    "overlap_text": overlap_text,
                    "overlap_character_count": len(overlap_text),
                },
                "tokenization": lab.tokenize(document.page_content),
                "vector": vector_details(reconstructed),
                "pca_2d": coordinates[faiss_position],
            }
        )

    return {
        "model": {
            "name": MODEL_NAME,
            "type": "Sentence Transformer bi-encoder",
            "base_encoder": "MiniLM transformer",
            "dimensions": lab.encoder.get_embedding_dimension(),
            "max_sequence_length": lab.encoder.max_seq_length,
            "purpose": "Encode chunks and questions independently into the same semantic vector space.",
        },
        "indexing_steps": [
            "RecursiveCharacterTextSplitter creates overlapping character-based chunks.",
            "The tokenizer converts every chunk into token IDs and an attention mask.",
            "MiniLM encodes tokens into contextual token representations.",
            "Sentence Transformer pooling creates one 384-dimensional vector per chunk.",
            "LangChain writes vectors into FAISS and text/metadata into its docstore.",
            "save_local writes index.faiss and index.pkl to disk.",
        ],
        "splitter": {
            "type": "RecursiveCharacterTextSplitter",
            "chunk_size_characters": active_configuration["chunk_size"],
            "chunk_overlap_characters": active_configuration["chunk_overlap"],
            "chunk_count": len(lab.chunks),
            "separators_tried_in_order": ["\n\n", "\n", " ", ""],
            "how_it_works": (
                "The splitter recursively tries paragraph breaks, line breaks, spaces, "
                "and finally individual characters until each chunk fits the size limit."
            ),
        },
        "original_document": "\n\n".join(item["text"] for item in documents),
        "faiss": {
            "index_class": type(index).__name__,
            "metric_type": "L2 distance" if index.metric_type == faiss.METRIC_L2 else str(index.metric_type),
            "dimensions": index.d,
            "total_vectors": index.ntotal,
            "meaning": "Lower L2 distance means the query vector is closer to a chunk vector.",
        },
        "records": records,
        "saved_files": inspect_saved_files(),
        "operation": {
            "name": request.operation,
            "message": operation_result["message"],
            "skipped_duplicates": operation_result["skipped_duplicates"],
        },
        "registry": {
            "has_existing_vector_database": lab.has_saved_index(),
            **registry,
        },
    }


@app.post("/api/search")
def search(request: SearchRequest):
    try:
        lab.ensure_index()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    query_vector = np.array(
        lab.embeddings.embed_query(request.question),
        dtype=np.float32,
    )
    distances, positions = lab.db.index.search(
        np.array([query_vector], dtype=np.float32),
        min(request.top_k, lab.db.index.ntotal),
    )

    results = []
    for rank, (position, distance) in enumerate(zip(positions[0], distances[0]), start=1):
        document_id = lab.db.index_to_docstore_id[int(position)]
        document = lab.db.docstore.search(document_id)
        chunk_vector = lab.db.index.reconstruct(int(position))
        results.append(
            {
                "rank": rank,
                "faiss_position": int(position),
                "document_id": document_id,
                "l2_distance": round(float(distance), 6),
                "manual_squared_l2": round(float(np.sum((query_vector - chunk_vector) ** 2)), 6),
                "text": document.page_content,
                "metadata": document.metadata,
                "chunk_vector_first_16": chunk_vector[:16].round(5).tolist(),
            }
        )

    relevant_results = [
        result for result in results
        if result["l2_distance"] <= request.maximum_l2_distance
    ]
    has_relevant_context = bool(relevant_results)

    return {
        "question": request.question,
        "tokenization": lab.tokenize(request.question),
        "query_vector": vector_details(query_vector),
        "retrieval_steps": [
            "Tokenize and encode the question with the same MiniLM sentence-transformer model.",
            "FAISS compares the query vector with stored chunk vectors using squared L2 distance.",
            "FAISS returns nearest vector positions and distances.",
            "LangChain maps FAISS positions to document IDs, then loads chunk text and metadata.",
            "The retrieved chunk text becomes context for an LLM in a complete RAG system.",
        ],
        "maximum_l2_distance": request.maximum_l2_distance,
        "nearest_neighbors": results,
        "results": relevant_results,
        "has_relevant_context": has_relevant_context,
        "retrieval_decision": (
            f"Accepted {len(relevant_results)} chunk(s) with squared L2 distance at or below "
            f"{request.maximum_l2_distance}."
            if has_relevant_context
            else (
                f"Rejected all nearest neighbors because none had squared L2 distance at or "
                f"below {request.maximum_l2_distance}."
            )
        ),
        "safe_answer": (
            "Relevant context was found. A complete RAG system may now call the LLM."
            if has_relevant_context
            else "I don't know based on the indexed document."
        ),
        "context_for_llm": "\n\n".join(result["text"] for result in relevant_results),
    }
