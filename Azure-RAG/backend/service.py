import os
from pathlib import Path
from typing import List, Optional

from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from load_llm import LoadLLM

BASE_DIR = Path(__file__).resolve().parent
DB_FAISS_PATH = BASE_DIR / "vectorstore" / "db_faiss"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class RagService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

    async def save_uploads(self, uploaded_files: List[UploadFile]) -> List[Path]:
        saved_paths = []

        for uploaded_file in uploaded_files:
            contents = await uploaded_file.read()
            safe_name = os.path.basename(uploaded_file.filename or "upload.pdf")
            destination = UPLOAD_DIR / safe_name
            destination.write_bytes(contents)
            saved_paths.append(destination)

        return saved_paths

    def load_pdf_documents(self, file_paths: List[Path]):
        documents = []

        for file_path in file_paths:
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())

        return documents

    def build_vector_store(self, documents):
        if not documents:
            raise ValueError("No documents available to build the vector store.")

        chunks = self.splitter.split_documents(documents)
        db = FAISS.from_documents(chunks, self.embeddings)
        db_dir = DB_FAISS_PATH.parent
        db_dir.mkdir(parents=True, exist_ok=True)
        db.save_local(str(DB_FAISS_PATH))
        return db

    def load_vector_store(self):
        if not DB_FAISS_PATH.exists() and not (DB_FAISS_PATH.with_suffix(".faiss")).exists():
            return None

        try:
            return FAISS.load_local(str(DB_FAISS_PATH), self.embeddings, allow_dangerous_deserialization=True)
        except TypeError:
            return FAISS.load_local(str(DB_FAISS_PATH), self.embeddings)

    def get_or_create_db(self, file_paths: Optional[List[Path]] = None):
        if file_paths:
            documents = self.load_pdf_documents(file_paths)
            return self.build_vector_store(documents)

        existing_db = self.load_vector_store()
        if existing_db is not None:
            return existing_db

        raise ValueError(
            "No vector store found. Upload at least one PDF file to create the knowledge base."
        )

    async def query(self, question: str, uploaded_files: Optional[List[UploadFile]] = None):
        if uploaded_files:
            file_paths = await self.save_uploads(uploaded_files)
        else:
            file_paths = []

        db = self.get_or_create_db(file_paths if file_paths else None)
        llm = LoadLLM.load_llm()

        mode_instruction = "You are an AI assistant. Use the uploaded documents to answer the question."

        documents = db.similarity_search(question, k=5)
        context = "\n\n".join([doc.page_content for doc in documents])

        llm_prompt = (
            f"{mode_instruction}\n\n"
            f"Use the following document excerpts to answer the question.\n\n"
            f"Documents:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer clearly and concisely."
        )

        response = llm.invoke(llm_prompt)
        answer_text = getattr(response, 'content', response) if response is not None else ''

        return {
            "question": question,
            "answer": answer_text,
        }
