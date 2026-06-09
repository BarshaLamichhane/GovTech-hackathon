import json
import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
from promptflow import tool


def _client():
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
def answer(question: str, context: str) -> str:
    documents = json.loads(context)
    formatted_context = "\n\n".join(
        f"Source: {doc['source']} | Page: {doc['page']}\n{doc['content']}"
        for doc in documents
    )
    response = _client().chat.completions.create(
        model=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the retrieved context. "
                    "If the context does not contain the answer, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{formatted_context}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content
