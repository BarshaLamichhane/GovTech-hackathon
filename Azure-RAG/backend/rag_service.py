import hashlib
import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import settings

BASE_DIR = Path(__file__).resolve().parent
DB_FAISS_PATH = BASE_DIR / "vectorstore" / "db_faiss"
DB_FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)


class LocalRagIndexService:
    def __init__(self):
        from langchain_huggingface import HuggingFaceEmbeddings

        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def build_vector_store(self, chunks: List[Document]):
        from langchain_community.vectorstores import FAISS

        db = FAISS.from_documents(chunks, self.embeddings)
        DB_FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
        db.save_local(str(DB_FAISS_PATH))
        return db

    def load_vector_store(self):
        from langchain_community.vectorstores import FAISS

        if not self.has_vector_store():
            return None
        try:
            return FAISS.load_local(
                str(DB_FAISS_PATH),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            return None

    def has_vector_store(self):
        return (
            DB_FAISS_PATH.exists()
            and (DB_FAISS_PATH / "index.faiss").exists()
            and (DB_FAISS_PATH / "index.pkl").exists()
        )


class AzureSearchRagIndexService:
    VECTOR_PROFILE_NAME = "govtech-vector-profile"
    VECTOR_ALGORITHM_NAME = "govtech-hnsw"

    def __init__(self):
        settings.validate_rag_provider()
        self.credential = self._build_credential()
        self.embeddings = self._build_embeddings()

    def _build_credential(self):
        if settings.azure_search_api_key:
            from azure.core.credentials import AzureKeyCredential

            return AzureKeyCredential(settings.azure_search_api_key)

        from azure.identity import DefaultAzureCredential

        return DefaultAzureCredential()

    def _build_embeddings(self):
        from langchain_openai import AzureOpenAIEmbeddings

        kwargs = {
            "azure_endpoint": settings.azure_openai_endpoint,
            "azure_deployment": settings.azure_openai_embedding_deployment,
            "api_version": settings.azure_openai_api_version,
            "dimensions": settings.azure_search_vector_dimensions,
        }
        if settings.azure_openai_api_key:
            kwargs["api_key"] = settings.azure_openai_api_key
        else:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default",
            )
        return AzureOpenAIEmbeddings(**kwargs)

    def _index_client(self):
        from azure.search.documents.indexes import SearchIndexClient

        return SearchIndexClient(
            endpoint=settings.azure_search_endpoint,
            credential=self.credential,
        )

    def _search_client(self):
        from azure.search.documents import SearchClient

        return SearchClient(
            endpoint=settings.azure_search_endpoint,
            index_name=settings.azure_search_index_name,
            credential=self.credential,
        )

    def _create_or_update_index(self):
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SearchableField,
            SimpleField,
            VectorSearch,
            VectorSearchProfile,
        )

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(
                name="source",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="metadata_json", type=SearchFieldDataType.String),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=settings.azure_search_vector_dimensions,
                vector_search_profile_name=self.VECTOR_PROFILE_NAME,
            ),
        ]
        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name=self.VECTOR_ALGORITHM_NAME)],
            profiles=[
                VectorSearchProfile(
                    name=self.VECTOR_PROFILE_NAME,
                    algorithm_configuration_name=self.VECTOR_ALGORITHM_NAME,
                )
            ],
        )
        index = SearchIndex(
            name=settings.azure_search_index_name,
            fields=fields,
            vector_search=vector_search,
        )
        self._index_client().create_or_update_index(index)

    def _replace_index(self):
        from azure.core.exceptions import ResourceNotFoundError

        index_client = self._index_client()
        try:
            index_client.delete_index(settings.azure_search_index_name)
        except ResourceNotFoundError:
            pass
        self._create_or_update_index()

    def build_vector_store(self, chunks: List[Document]):
        self._replace_index()
        texts = [chunk.page_content for chunk in chunks]
        vectors = self.embeddings.embed_documents(texts)
        documents = []

        for chunk, vector in zip(chunks, vectors):
            metadata = dict(chunk.metadata)
            source = str(metadata.get("source", "unknown"))
            page = metadata.get("page")
            content_hash = hashlib.sha256(
                f"{source}:{page}:{chunk.page_content}".encode("utf-8")
            ).hexdigest()
            documents.append(
                {
                    "id": content_hash,
                    "content": chunk.page_content,
                    "source": source,
                    "page": page if isinstance(page, int) else -1,
                    "metadata_json": json.dumps(metadata, default=str),
                    "content_vector": vector,
                }
            )

        search_client = self._search_client()
        for offset in range(0, len(documents), 500):
            results = search_client.upload_documents(documents=documents[offset:offset + 500])
            failures = [result for result in results if not result.succeeded]
            if failures:
                raise RuntimeError(f"Azure AI Search failed to index {len(failures)} chunks.")

        return self

    def load_vector_store(self):
        return self if self.has_vector_store() else None

    def has_vector_store(self):
        from azure.core.exceptions import ResourceNotFoundError

        try:
            self._index_client().get_index(settings.azure_search_index_name)
            return True
        except ResourceNotFoundError:
            return False

    def similarity_search(self, query: str, k: int = 5):
        from azure.search.documents.models import VectorizedQuery

        vector_query = VectorizedQuery(
            vector=self.embeddings.embed_query(query),
            k_nearest_neighbors=k,
            fields="content_vector",
            kind="vector",
        )
        results = self._search_client().search(
            search_text=query,
            vector_queries=[vector_query],
            select=["content", "source", "page", "metadata_json"],
            top=k,
        )

        documents = []
        for result in results:
            metadata = json.loads(result.get("metadata_json") or "{}")
            metadata["source"] = result.get("source", metadata.get("source", "unknown"))
            metadata["page"] = result.get("page", metadata.get("page"))
            metadata["score"] = result.get("@search.score")
            documents.append(Document(page_content=result["content"], metadata=metadata))
        return documents


class RagIndexService:
    def __init__(self):
        settings.validate_rag_provider()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        self.provider = (
            AzureSearchRagIndexService()
            if settings.rag_provider == "azure"
            else LocalRagIndexService()
        )

    def build_vector_store(self, documents: List[Document]):
        if not documents:
            raise ValueError("No documents available to build the vector store.")
        return self.provider.build_vector_store(self.splitter.split_documents(documents))

    def load_vector_store(self):
        return self.provider.load_vector_store()

    def has_vector_store(self):
        return self.provider.has_vector_store()

    def update_vector_store(self, documents: List[Document]):
        return self.build_vector_store(documents)
