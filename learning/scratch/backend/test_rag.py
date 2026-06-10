from fastapi.testclient import TestClient

import main
from main import app

client = TestClient(app)


def test_index_and_search():
    index_response = client.post(
        "/api/index",
        json={
            "text": "Solar panels create electricity from sunlight. Batteries store electricity for later.",
            "chunk_size": 6,
            "chunk_overlap": 1,
        },
    )
    assert index_response.status_code == 200
    assert len(index_response.json()["chunks"]) == 2
    assert index_response.json()["chunks"][1]["overlap_from_previous"] == ["sunlight."]
    assert len(index_response.json()["vector_store_records"]) == 2

    search_response = client.post(
        "/api/search",
        json={"question": "What do batteries store?", "top_k": 1},
    )
    assert search_response.status_code == 200
    assert "Batteries" in search_response.json()["retrieved_chunks"][0]["text"]
    assert search_response.json()["question_tokens"] == ["what", "do", "batteries", "store"]
    assert "dot_product" in search_response.json()["all_similarity_scores"][0]
    assert search_response.json()["has_relevant_context"] is True

    irrelevant_response = client.post(
        "/api/search",
        json={
            "question": "Who won the football championship?",
            "top_k": 1,
            "minimum_similarity": 0.2,
        },
    )
    assert irrelevant_response.status_code == 200
    assert irrelevant_response.json()["nearest_neighbors"]
    assert irrelevant_response.json()["retrieved_chunks"] == []
    assert irrelevant_response.json()["has_relevant_context"] is False
    assert irrelevant_response.json()["context_sent_to_llm"] == ""
    assert irrelevant_response.json()["simple_extractive_answer"].startswith("I don't know")


def test_overlap_must_be_smaller_than_chunk_size():
    response = client.post(
        "/api/index",
        json={"text": "one two three four five six", "chunk_size": 5, "chunk_overlap": 5},
    )
    assert response.status_code == 400


def test_multiple_pdf_uploads_populate_editable_text(monkeypatch):
    monkeypatch.setattr(
        main,
        "extract_pdf_text",
        lambda contents: {
            "page_count": 2,
            "pages_with_text": 2,
            "text": "Extracted PDF learning text.",
            "page_previews": [{"page": 1, "preview": "Extracted PDF learning text."}],
        },
    )
    response = client.post(
        "/api/extract-pdfs",
        files=[
            ("files", ("first.pdf", b"first-fake-pdf", "application/pdf")),
            ("files", ("second.pdf", b"second-fake-pdf", "application/pdf")),
        ],
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["file_count"] == 2
    assert payload["total_pages"] == 4
    assert "=== DOCUMENT: first.pdf ===" in payload["combined_text"]
    assert "=== DOCUMENT: second.pdf ===" in payload["combined_text"]
