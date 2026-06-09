# Azure-RAG Backend

The same API can run with local infrastructure or Azure services.

| Mode | Generation | Retrieval and vectors |
| --- | --- | --- |
| Fully local | GGUF model through LlamaCpp | Hugging Face embeddings and FAISS |
| Azure | Azure OpenAI chat and embeddings | Azure AI Search hybrid vector search |
| Mixed | Local, Mistral, or Gemini generation | Local FAISS or Azure AI Search |

The frontend and API routes are identical in every mode.

## Install

```bash
cd Azure-RAG/backend
python3 -m pip install -r requirements.txt
cp .env.example .env
```

Start the API:

```bash
uvicorn main:app --reload --port 8000
```

`GET /api/health` reports the selected RAG and LLM providers.

## Local Mode

Use FAISS with a local GGUF model:

```env
RAG_PROVIDER=local
LLM_PROVIDER=local
OFFLINE_MODEL_PATH=/absolute/path/to/model.gguf
```

You can also keep `RAG_PROVIDER=local` and select `mistral`, `gemini`, or `azure` as the LLM provider.

## Azure Mode

Create these Azure resources:

1. An Azure OpenAI resource with one chat deployment and one embedding deployment.
2. An Azure AI Search service.

Configure `.env`:

```env
RAG_PROVIDER=azure
LLM_PROVIDER=azure

AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small

AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_API_KEY=your-key
AZURE_SEARCH_INDEX_NAME=govtech-rag-index
AZURE_SEARCH_VECTOR_DIMENSIONS=1536
```

`AZURE_SEARCH_VECTOR_DIMENSIONS` must match the embedding deployment. The default `1536` matches `text-embedding-3-small`.

When PDFs are uploaded, the backend:

1. Extracts and chunks PDF text.
2. Creates embeddings with the Azure OpenAI embedding deployment.
3. Replaces and populates the configured Azure AI Search index.
4. Runs hybrid keyword and vector retrieval.
5. Sends retrieved context to the configured LLM.

This application-managed ingestion is intentionally simpler than the Blob Storage indexer and skillset pipeline in Microsoft's `azure-search-classic-rag` notebook. It keeps browser PDF uploads and local/Azure behavior aligned. The Microsoft indexer pipeline can later replace ingestion without changing the query schema.

## Keyless Azure Authentication

Leave `AZURE_OPENAI_API_KEY` and `AZURE_SEARCH_API_KEY` empty to use `DefaultAzureCredential`. For local development, run:

```bash
az login
```

Assign the identity:

- `Cognitive Services OpenAI User`
- `Search Service Contributor`
- `Search Index Data Contributor`
- `Search Index Data Reader`

Use a managed identity with the same roles when deploying the backend to Azure.

## Prompt Flow

The flow in `../prompt-flow/rag-chat` uses the same Azure AI Search index and Azure OpenAI deployments:

```bash
cd Azure-RAG/prompt-flow/rag-chat
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
pf flow test --flow . --inputs question="What are the main risks?"
```

Use a separate environment for Prompt Flow because its orchestration dependencies can conflict with application dependencies such as the Mistral client. Set the Azure environment variables before running the flow. The folder can also be uploaded into Azure Machine Learning Prompt Flow. In Azure, store secrets in managed connections or use managed identity instead of plain environment variables.

The flow contains two nodes:

- `retrieve`: hybrid keyword and vector retrieval from Azure AI Search.
- `answer`: grounded answer generation through Azure OpenAI.

## API Endpoints

- `GET /api/health`
- `GET /api/knowledge-base/status`
- `POST /api/chat` with `question` and zero or more `files`
- `POST /api/build-knowledge-base` with one or more `files`
