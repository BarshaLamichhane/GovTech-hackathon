from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from service import RagService
from settings import settings

app = FastAPI(title="Azure RAG Chat Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

service = RagService()


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "rag_provider": settings.rag_provider,
        "llm_provider": settings.llm_provider,
    }


@app.post("/api/chat")
async def chat(
    question: str = Form(...),
    files: List[UploadFile] = File(default=[]),
):
    try:
        response = await service.query(question=question, uploaded_files=files)
        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")


@app.post("/api/build-knowledge-base")
async def build_knowledge_base(
    files: List[UploadFile] = File(default=[]),
):
    try:
        response = await service.create_knowledge_base(uploaded_files=files)
        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")


@app.get("/api/knowledge-base/status")
def knowledge_base_status():
    return {
        "has_knowledge_base": service.has_knowledge_base(),
        "has_vector_database": service.has_vector_store(),
        "generated_pages": service.get_knowledge_page_names(),
        "rag_provider": settings.rag_provider,
    }
