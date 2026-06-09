from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.service import RagService

app = FastAPI(title="GovTech Screening Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

service = RagService()


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/modes")
def get_modes():
    return {
        "modes": [
            "Chat with reports",
            "AI Project Readiness Advisor",
            "Risk Dashboard",
        ]
    }


@app.post("/api/chat")
async def chat(
    question: str = Form(...),
    mode: str = Form("Chat with reports"),
    files: List[UploadFile] = File(default=[]),
):
    """Accept a question and optional PDF uploads, then return a RAG response."""
    response = await service.query(question=question, mode=mode, uploaded_files=files)
    return response
