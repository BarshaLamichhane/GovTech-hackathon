import json
import os

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
from promptflow import tool


def _search_credential():
    api_key = os.getenv("AZURE_SEARCH_API_KEY")
    return AzureKeyCredential(api_key) if api_key else DefaultAzureCredential()


def _openai_client():
    kwargs = {
        "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    }
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    else:
        kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default",
        )
    return AzureOpenAI(**kwargs)


@tool
def retrieve(question: str) -> str:
    embedding = _openai_client().embeddings.create(
        model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        input=question,
    ).data[0].embedding
    vector_query = VectorizedQuery(
        vector=embedding,
        k_nearest_neighbors=5,
        fields="content_vector",
        kind="vector",
    )
    search_client = SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=os.getenv("AZURE_SEARCH_INDEX_NAME", "govtech-rag-index"),
        credential=_search_credential(),
    )
    results = search_client.search(
        search_text=question,
        vector_queries=[vector_query],
        select=["content", "source", "page"],
        top=5,
    )
    return json.dumps(
        [
            {
                "content": result["content"],
                "source": result.get("source", "unknown"),
                "page": result.get("page"),
            }
            for result in results
        ]
    )
