# Backend Service

This folder contains the Python backend that exposes a REST API for the GovTech screening demo.

## Run locally

1. Create a virtual environment:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the API server:

```bash
uvicorn main:app --reload --port 8000
```

## API Endpoints

- `GET /api/health` - health check
- `GET /api/modes` - available UI modes
- `POST /api/chat` - ask a question with optional PDF uploads

The backend uses the existing LangChain and LLM configuration from the root repository.
