import json
import os
import re
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)


class WikiService:
    def __init__(self):
        pass

    def slugify(self, text: str):
        slug = os.path.splitext(text or "untitled")[0].lower()
        slug = re.sub(r"[^a-z0-9]+", "-", slug)
        slug = re.sub(r"-+", "-", slug).strip("-")
        return slug or "untitled"

    def load_markdown_documents(self):
        documents = []
        for md_path in sorted(KNOWLEDGE_DIR.glob("*.md")):
            content = md_path.read_text(encoding="utf-8")
            documents.append(Document(page_content=content, metadata={"source": str(md_path.name)}))
        return documents

    def build_wiki_pages(self, file_paths: List[Path], llm=None):
        # Simple wrapper that extracts text from PDFs and delegates JSON extraction to provided LLM
        source_texts = []
        for file_path in file_paths:
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            for doc in docs:
                page = doc.metadata.get("page", "unknown")
                text = doc.page_content.strip()
                if not text:
                    continue
                source_texts.append({
                    "source": file_path.name,
                    "page": page,
                    "text": text[:4000],
                })

        if not source_texts:
            raise ValueError("No readable PDF text found in uploaded files.")

        document_context = "\n\n".join([
            f"SOURCE: {item['source']} | PAGE: {item['page']}\n{item['text']}" for item in source_texts
        ])

        prompt = f"""
You are a knowledge base author.
Create an entity-centric linked wiki from the following document text.
Extract the most important entities, define each entity, and capture relationships between entities.
Output valid JSON with one root object containing the key "entities".
Each entity must include: name, type, slug, summary, related_entities, sources, key_facts.
If the input is in English, keep the output in English.

Document text:
{document_context}
"""

        # fallback entity if LLM is not provided or fails
        fallback_entity = {
            "name": file_paths[0].stem if file_paths else "entity",
            "type": "Document",
            "slug": self.slugify(file_paths[0].stem if file_paths else "entity"),
            "summary": f"Entity knowledge generated from uploaded documents.",
            "related_entities": [],
            "sources": [file_paths[0].name if file_paths else "unknown"],
            "key_facts": ["Entity-centric knowledge page generated."],
        }

        entities = None
        if llm is not None:
            try:
                response = llm.invoke(prompt)
                text = getattr(response, 'content', response) or ''
                text = text.strip().replace("```json", "").replace("```", "").strip()
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    entities = parsed.get('entities') or []
                elif isinstance(parsed, list):
                    entities = parsed
            except Exception:
                entities = [fallback_entity]

        if not entities:
            entities = [fallback_entity]

        generated_pages = []
        for entity in entities:
            name = entity.get("name") or "Unnamed Entity"
            slug = self.slugify(entity.get("slug") or name)
            title = name
            entity_type = entity.get("type", "entity")
            summary = entity.get("summary", "No summary available.")
            sources = entity.get("sources", []) or []
            key_facts = entity.get("key_facts", []) or []
            related_entities = entity.get("related_entities", []) or []

            page_path = KNOWLEDGE_DIR / f"{slug}.md"
            lines = [
                "---",
                f"title: {title}",
                f"slug: {slug}",
                f"type: entity",
                f"entity_type: {entity_type}",
                "---",
                "",
                f"# {title}",
                "",
                f"**Type:** {entity_type}",
                "",
                "**Sources:**",
                "",
            ]
            for source in sources:
                source_text = source.strip()
                if source_text:
                    lines.append(f"- {source_text}")
            lines.extend([
                "",
                "## Summary",
                "",
                summary.strip(),
                "",
                "## Key Facts",
                "",
            ])
            for fact in key_facts:
                lines.append(f"- {fact}")
            lines.extend(["", "## Related Entities", ""]) 
            for related in related_entities:
                related_name = related.strip()
                if related_name:
                    related_slug = self.slugify(related_name)
                    lines.append(f"- [{related_name}]({related_slug}.md)")

            page_path.write_text("\n".join(lines), encoding="utf-8")
            generated_pages.append(str(page_path.name))

        return generated_pages
