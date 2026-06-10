import hashlib

import pytest
from fastapi.testclient import TestClient

import main
from main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_vector_database(monkeypatch, tmp_path):
    db_path = tmp_path / "vectorstore_demo"
    monkeypatch.setattr(main, "DB_PATH", db_path)
    monkeypatch.setattr(main, "REGISTRY_PATH", db_path / "registry.json")
    main.lab.db = None
    main.lab.chunks = []


def test_real_embedding_and_faiss_flow():
    indexed = client.post(
        "/api/index",
        json={
            "text": (
                "Solar panels transform sunlight into electricity. "
                "Batteries preserve energy for use at night. "
                "A vector database retrieves semantically related text."
            ),
            "chunk_size": 55,
            "chunk_overlap": 10,
        },
    )
    assert indexed.status_code == 200
    payload = indexed.json()
    assert payload["model"]["dimensions"] == 384
    assert payload["faiss"]["total_vectors"] >= 2
    assert payload["records"][0]["tokenization"]["token_ids"]
    assert payload["records"][0]["chunking"]["start_character"] == 0
    assert payload["records"][0]["chunking"]["character_count"] <= 55
    assert payload["splitter"]["separators_tried_in_order"] == ["\n\n", "\n", " ", ""]
    assert payload["saved_files"]["index_faiss"]["size_bytes"] > 0
    assert payload["saved_files"]["index_pkl"]["size_bytes"] > 0
    assert payload["saved_files"]["registry_json"]["size_bytes"] > 0
    assert payload["saved_files"]["index_faiss"]["details"]["dimensions_per_vector"] == 384
    assert payload["saved_files"]["index_faiss"]["details"]["sample_vectors"]
    assert payload["saved_files"]["index_pkl"]["details"]["documents"][0]["text"]
    assert payload["saved_files"]["registry_json"]["details"]["documents"]
    assert payload["operation"]["name"] == "recreate"

    searched = client.post(
        "/api/search",
        json={"question": "How can energy be kept until nighttime?", "top_k": 1},
    )
    assert searched.status_code == 200
    search_payload = searched.json()
    assert "Batteries" in search_payload["results"][0]["text"]
    assert search_payload["results"][0]["l2_distance"] == search_payload["results"][0]["manual_squared_l2"]
    assert search_payload["has_relevant_context"] is True

    irrelevant = client.post(
        "/api/search",
        json={
            "question": "Who won the football world cup?",
            "top_k": 2,
            "maximum_l2_distance": 1.2,
        },
    )
    assert irrelevant.status_code == 200
    irrelevant_payload = irrelevant.json()
    assert irrelevant_payload["nearest_neighbors"]
    assert irrelevant_payload["results"] == []
    assert irrelevant_payload["has_relevant_context"] is False
    assert irrelevant_payload["context_for_llm"] == ""
    assert irrelevant_payload["safe_answer"].startswith("I don't know")


def test_multiple_pdf_uploads_populate_editable_text(monkeypatch):
    monkeypatch.setattr(
        main,
        "extract_pdf_text",
        lambda contents: {
            "page_count": 1,
            "pages_with_text": 1,
            "text": "Model-powered PDF text.",
            "page_previews": [{"page": 1, "preview": "Model-powered PDF text."}],
        },
    )
    response = client.post(
        "/api/extract-pdfs",
        files=[
            ("files", ("model-one.pdf", b"first-fake-pdf", "application/pdf")),
            ("files", ("model-two.pdf", b"second-fake-pdf", "application/pdf")),
        ],
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["file_count"] == 2
    assert payload["total_pages"] == 2
    assert "=== DOCUMENT: model-one.pdf ===" in payload["combined_text"]
    assert "=== DOCUMENT: model-two.pdf ===" in payload["combined_text"]
    assert payload["files"][0]["file_hash"] == hashlib.sha256(b"first-fake-pdf").hexdigest()


def test_index_lifecycle_registry_and_recommendations():
    first_document = {
        "filename": "energy.pdf",
        "text": "Solar panels create electricity. Batteries store energy for use at night.",
        "file_hash": "energy-v1",
    }
    second_document = {
        "filename": "wind.pdf",
        "text": "Wind turbines convert moving air into electricity.",
        "file_hash": "wind-v1",
    }
    config = {"chunk_size": 55, "chunk_overlap": 10}

    recreated = client.post(
        "/api/index",
        json={"operation": "recreate", "documents": [first_document], **config},
    )
    assert recreated.status_code == 200
    recreated_payload = recreated.json()
    assert recreated_payload["registry"]["documents"][0]["filename"] == "energy.pdf"
    assert recreated_payload["registry"]["documents"][0]["file_hash"] == "energy-v1"
    assert recreated_payload["registry"]["documents"][0]["chunk_count"] >= 1
    assert recreated_payload["registry"]["documents"][0]["indexed_at"]

    same = client.post(
        "/api/index/recommendation",
        json={"documents": [first_document], **config},
    ).json()
    assert same["recommended_operation"] == "use_existing"

    duplicate_update = client.post(
        "/api/index",
        json={"operation": "update", "documents": [first_document], **config},
    )
    assert duplicate_update.status_code == 200
    assert duplicate_update.json()["operation"]["skipped_duplicates"] == ["energy.pdf"]

    new_document = client.post(
        "/api/index/recommendation",
        json={"documents": [first_document, second_document], **config},
    ).json()
    assert new_document["recommended_operation"] == "update"
    assert new_document["new_documents"] == ["wind.pdf"]

    updated = client.post(
        "/api/index",
        json={
            "operation": "update",
            "documents": [first_document, second_document, {**second_document, "filename": "wind-copy.pdf"}],
            **config,
        },
    )
    assert updated.status_code == 200
    assert len(updated.json()["registry"]["documents"]) == 2
    assert updated.json()["operation"]["skipped_duplicates"] == ["wind-copy.pdf", "energy.pdf"]

    changed_config = client.post(
        "/api/index/recommendation",
        json={"documents": [first_document], "chunk_size": 70, "chunk_overlap": 10},
    ).json()
    assert changed_config["recommended_operation"] == "recreate"

    rejected_update = client.post(
        "/api/index",
        json={
            "operation": "update",
            "documents": [second_document],
            "chunk_size": 70,
            "chunk_overlap": 10,
        },
    )
    assert rejected_update.status_code == 400
    assert "configuration changed for chunk_size" in rejected_update.json()["detail"]

    main.lab.db = None
    loaded = client.post("/api/index", json={"operation": "use_existing", **config})
    assert loaded.status_code == 200
    assert loaded.json()["operation"]["name"] == "use_existing"
    assert len(loaded.json()["registry"]["documents"]) == 2

    changed_first_document = {
        **first_document,
        "text": "Solar panels and batteries have updated documentation.",
        "file_hash": "energy-v2",
    }
    changed = client.post(
        "/api/index",
        json={"operation": "update", "documents": [changed_first_document], **config},
    )
    assert changed.status_code == 200
    registry_by_name = {
        item["filename"]: item for item in changed.json()["registry"]["documents"]
    }
    assert registry_by_name["energy.pdf"]["file_hash"] == "energy-v2"
    assert len(registry_by_name) == 2


def test_default_text_update_uses_matching_default_configuration():
    default_text = (
        "Solar panels transform sunlight into electricity. They are most productive during bright daylight. "
        "Batteries preserve extra solar energy for use at night.\n\n"
        "Wind turbines convert moving air into electricity. Wind and solar power are renewable energy sources."
    )

    recreated = client.post(
        "/api/index",
        json={"operation": "recreate", "text": default_text},
    )
    assert recreated.status_code == 200
    assert recreated.json()["registry"]["configuration"]["chunk_size"] == 220
    assert recreated.json()["registry"]["configuration"]["chunk_overlap"] == 40

    recommendation = client.post(
        "/api/index/recommendation",
        json={"text": default_text},
    )
    assert recommendation.status_code == 200
    assert recommendation.json()["recommended_operation"] == "use_existing"

    unchanged_update = client.post(
        "/api/index",
        json={"operation": "update", "text": default_text},
    )
    assert unchanged_update.status_code == 200
    assert unchanged_update.json()["operation"]["skipped_duplicates"] == [
        "pasted-or-default-text.txt"
    ]


def test_legacy_index_without_registry_explains_one_time_recreate(monkeypatch):
    monkeypatch.setattr(main.ModelRagLab, "has_saved_index", lambda self: True)

    recommendation = client.post(
        "/api/index/recommendation",
        json={"text": "Default text using default configuration."},
    )
    assert recommendation.status_code == 200
    assert recommendation.json()["recommended_operation"] == "recreate"
    assert "created before registry tracking" in recommendation.json()["reason"]

    update = client.post(
        "/api/index",
        json={"operation": "update", "text": "Default text using default configuration."},
    )
    assert update.status_code == 400
    assert "no document registry or saved configuration" in update.json()["detail"]
