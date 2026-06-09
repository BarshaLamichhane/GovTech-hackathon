# Azure-RAG Backend

This backend provides a REST API for the Azure-RAG chatbot using multiple PDF uploads and a retrieval-augmented generation workflow.

## Setup

1. Activate the existing project virtual environment:

```bash
cd Azure-RAG/backend
source ../../venv/bin/activate
```

2. Install dependencies into the existing environment:

```bash
python3 -m pip install -r requirements.txt
```

If you prefer to keep the backend isolated, you can still create a dedicated environment inside `Azure-RAG/backend`, but the recommended flow is to reuse the shared `govtech-hackathon/venv` environment.

> If you only need offline model support, you can keep the online model packages in `requirements.txt` but do not need to configure `MISTRAL_API_KEY` or `GEMINI_API_KEY`.

3. Create a `.env` file in `Azure-RAG/backend` with either offline or online model settings:

```env
# For offline use:
OFFLINE_MODEL_PATH=/path/to/your/model.gguf

# For online use with Mistral:
MISTRAL_API_KEY=your_api_key
MISTRAL_API_MODEL_NAME=mistral-large-latest

# Or for Gemini:
GEMINI_API_KEY=your_gemini_api_key
GEMINI_API_MODEL_NAME=gemini-1.5-pro
```

4. Start the API server backend:

```bash
uvicorn main:app --reload --port 8000
```

## API endpoints

- `GET /api/health`
- `POST /api/chat` - accepts `question` and multiple `files`

> Note: The current backend requires at least one uploaded PDF or an existing FAISS index before answering chat questions.

5. Start the frontend

```bash
cd /Users/barshalamichhane/Documents/python-project/LLM-Projects/GovTech-hackathon/Azure-RAG/frontend
npm install
npm run dev
```
