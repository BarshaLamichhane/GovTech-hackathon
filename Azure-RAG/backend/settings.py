import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    rag_provider: str = _env("RAG_PROVIDER", "local").lower()
    llm_provider: str = _env("LLM_PROVIDER", "auto").lower()

    azure_openai_endpoint: str = _env("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str = _env("AZURE_OPENAI_API_KEY")
    azure_openai_api_version: str = _env("AZURE_OPENAI_API_VERSION", "2024-10-21")
    azure_openai_chat_deployment: str = _env("AZURE_OPENAI_CHAT_DEPLOYMENT")
    azure_openai_embedding_deployment: str = _env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    azure_search_endpoint: str = _env("AZURE_SEARCH_ENDPOINT")
    azure_search_api_key: str = _env("AZURE_SEARCH_API_KEY")
    azure_search_index_name: str = _env("AZURE_SEARCH_INDEX_NAME", "govtech-rag-index")
    azure_search_vector_dimensions: int = int(_env("AZURE_SEARCH_VECTOR_DIMENSIONS", "1536"))

    def validate_rag_provider(self) -> None:
        if self.rag_provider not in {"local", "azure"}:
            raise RuntimeError("RAG_PROVIDER must be either 'local' or 'azure'.")

        if self.rag_provider == "azure":
            missing = [
                name
                for name, value in {
                    "AZURE_SEARCH_ENDPOINT": self.azure_search_endpoint,
                    "AZURE_OPENAI_ENDPOINT": self.azure_openai_endpoint,
                    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": self.azure_openai_embedding_deployment,
                }.items()
                if not value
            ]
            if missing:
                raise RuntimeError(f"Azure RAG configuration is missing: {', '.join(missing)}")


settings = Settings()
