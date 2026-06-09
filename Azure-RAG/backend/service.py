import json
import os
import re
from pathlib import Path
from typing import List, Optional

from fastapi import UploadFile
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from load_llm import LoadLLM
from wiki_service import WikiService
from rag_service import RagIndexService

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class RagService:
    def __init__(self):
        self.wiki = WikiService()
        self.rag = RagIndexService()

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
            pdf_docs = loader.load()
            for doc in pdf_docs:
                doc.metadata["source"] = file_path.name
            documents.extend(pdf_docs)
        return documents

    def load_markdown_documents(self):
        return self.wiki.load_markdown_documents()

    def build_vector_store(self, documents):
        return self.rag.build_vector_store(documents)

    def extract_json_from_response(self, text: str):
        text = text.strip()
        text = text.replace("```json", "").replace("```", "").strip()

        if not text:
            raise ValueError("Empty response from LLM")

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start_candidates = [text.find("{"), text.find("[")]
            start = min([pos for pos in start_candidates if pos != -1], default=-1)
            end_candidates = [text.rfind("}"), text.rfind("]")]
            end = max([pos for pos in end_candidates if pos != -1], default=-1)
            if start == -1 or end == -1:
                raise ValueError("No JSON structure found in LLM response")
            return json.loads(text[start:end + 1])

    def slugify(self, text: str):
        return self.wiki.slugify(text)

    def write_markdown_page(self, slug: str, title: str, summary: str, topics: List[str], key_points: List[str]):
        # convenience wrapper — not currently used elsewhere
        return self.wiki.build_wiki_pages([])

    def build_wiki_pages(self, file_paths: List[Path]):
        llm = LoadLLM.load_llm()
        return self.wiki.build_wiki_pages(file_paths, llm=llm)

    def load_vector_store(self):
        return self.rag.load_vector_store()

    def has_vector_store(self):
        return self.rag.has_vector_store()

    def has_knowledge_base(self):
        return bool(self.wiki.load_markdown_documents())

    def get_knowledge_page_names(self):
        return [doc.metadata.get('source') for doc in self.wiki.load_markdown_documents()]

    def update_vector_store(self, documents):
        return self.rag.update_vector_store(documents)

    def get_or_create_db(self, file_paths: Optional[List[Path]] = None):
        existing_db = self.load_vector_store()

        if file_paths:
            pdf_documents = self.load_pdf_documents(file_paths)
            return self.update_vector_store(pdf_documents)

        if existing_db is not None:
            return existing_db

        knowledge_documents = self.load_markdown_documents()
        if knowledge_documents:
            return self.build_vector_store(knowledge_documents)

        raise ValueError(
            "No document index found. Upload at least one PDF in RAG Chat or create a Wiki LLM."
        )

    async def create_knowledge_base(self, uploaded_files: Optional[List[UploadFile]] = None):
        if not uploaded_files:
            if self.has_vector_store():
                return {
                    "message": "Existing knowledge base and vector index found.",
                    "generated_pages": self.get_knowledge_page_names(),
                }
            raise ValueError("Upload at least one PDF to create the knowledge base.")

        file_paths = await self.save_uploads(uploaded_files)
        page_names = self.build_wiki_pages(file_paths)
        knowledge_documents = self.load_markdown_documents()

        if self.has_vector_store():
            self.update_vector_store(knowledge_documents)
            message = "Entity-centric knowledge base updated and vector database rebuilt with new documents."
        else:
            self.build_vector_store(knowledge_documents)
            message = "Entity-centric knowledge base created successfully."

        return {
            "message": message,
            "generated_pages": page_names,
        }

    async def query(self, question: str, uploaded_files: Optional[List[UploadFile]] = None):
        if uploaded_files:
            file_paths = await self.save_uploads(uploaded_files)
        else:
            file_paths = []

        db = self.get_or_create_db(file_paths if file_paths else None)
        llm = LoadLLM.load_llm()

        mode_instruction = (
            "You are an AI assistant. Answer using only the retrieved document context. "
            "If the context does not contain the answer, say so."
        )
        documents = db.similarity_search(question, k=5)
        context = "\n\n".join([f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}" for doc in documents])

        llm_prompt = (
            f"{mode_instruction}\n\n"
            f"Use the following document excerpts to answer the question:\n\n"
            f"{context}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer clearly and concisely."
        )

        response = llm.invoke(llm_prompt)
        answer_text = getattr(response, 'content', response) if response is not None else ''

        return {
            "question": question,
            "answer": answer_text,
        }
