# Prompt Flow

`rag-chat` is a deployable Prompt Flow version of the Azure RAG query path.

It expects an Azure AI Search index created by the backend in Azure mode. This keeps ingestion in the application while Prompt Flow handles orchestration, testing, evaluation, and deployment of the retrieval and generation steps.

Run it in a dedicated virtual environment because Prompt Flow's orchestration dependencies can conflict with backend provider packages:

```bash
cd Azure-RAG/prompt-flow/rag-chat
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Required environment variables:

```env
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_CHAT_DEPLOYMENT=
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=
AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_API_KEY=
AZURE_SEARCH_INDEX_NAME=govtech-rag-index
```

API keys are optional when the runtime identity has the required Azure RBAC roles.
