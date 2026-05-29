import os
import json
import re
import tempfile
from datetime import date

import altair as alt
import pandas as pd
import streamlit as st
from streamlit_chat import message

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from loadllm import Loadllm


DB_FAISS_PATH = "vectorstore/db_faiss"
GOVTECH_HUB_DIR = "external/govtech-sandbox-hub"
GOVTECH_HUB_WIKI_DIR = os.path.join(GOVTECH_HUB_DIR, "wiki")
GOVTECH_HUB_REPORTS_DIR = os.path.join(GOVTECH_HUB_DIR, "context", "reports", "en")

# Local generated wiki plus likely locations if the GitHub repo is cloned under external/.
WIKI_SOURCE_DIRS = [
    "wiki_pages",
    "external/govtech-sandbox-hub/wiki_pages",
    "external/govtech-sandbox-hub/llm_wiki",
    "external/govtech-sandbox-hub/wiki",
    "external/govtech-sandbox-hub/docs/wiki_pages",
    "external/govtech-sandbox-hub/docs/llm_wiki",
]

WIKI_FILE_EXTENSIONS = {".md", ".markdown", ".txt"}

# Drop MIT AI Risk Repository and NIST AI RMF files into these folders.
EXTERNAL_RISK_SOURCE_DIRS = [
    "external/risk_sources",
    "external/risk_sources/mit_ai_risk_repository",
    "external/risk_sources/nist_ai_rmf",
    "external/govtech-sandbox-hub/risk_sources",
    "external/govtech-sandbox-hub/external_sources",
]

EXTERNAL_RISK_FILE_EXTENSIONS = {
    ".md",
    ".markdown",
    ".txt",
    ".csv",
    ".json",
    ".pdf",
}

MIT_RISK_DOMAINS = [
    {
        "domain": "Discrimination & Toxicity",
        "subdomain": "Unfair discrimination and misrepresentation",
        "risk": "Unequal treatment or representation of individuals or groups by an AI system.",
        "timing": "pilot-design",
        "pilot_relevance": "Check training data, evaluation data, and decision criteria before pilot launch.",
        "mitigation": "Review representativeness of data, define fairness checks, and test outputs with affected user groups.",
    },
    {
        "domain": "Discrimination & Toxicity",
        "subdomain": "Unequal performance across groups",
        "risk": "AI accuracy or effectiveness differs across groups or user contexts.",
        "timing": "pilot-evaluation",
        "pilot_relevance": "Measure performance across relevant groups and operating conditions.",
        "mitigation": "Run subgroup evaluation, document limitations, and set minimum acceptance thresholds before rollout.",
    },
    {
        "domain": "Privacy & Security",
        "subdomain": "Compromise of privacy",
        "risk": "Sensitive information is exposed, inferred, memorized, or shared without authorization.",
        "timing": "pilot-design",
        "pilot_relevance": "Map personal data flows, minimize data, and validate access controls.",
        "mitigation": "Apply data minimization, access control, privacy review, logging policy, and secure retention rules.",
    },
    {
        "domain": "Privacy & Security",
        "subdomain": "AI system security vulnerabilities and attacks",
        "risk": "The AI system, integrations, or toolchain can be attacked or manipulated.",
        "timing": "pilot-operation",
        "pilot_relevance": "Assess prompt injection, model abuse, logging, secrets, and integration security.",
        "mitigation": "Threat-model integrations, restrict tools and secrets, monitor misuse, and test prompt-injection scenarios.",
    },
    {
        "domain": "Misinformation",
        "subdomain": "False or misleading information",
        "risk": "The AI system generates or spreads incorrect or misleading outputs.",
        "timing": "pilot-operation",
        "pilot_relevance": "Define human review, quality thresholds, and escalation rules.",
        "mitigation": "Use human review, source citation, confidence thresholds, red-team examples, and correction workflows.",
    },
    {
        "domain": "Malicious Actors",
        "subdomain": "Fraud, scams, and targeted manipulation",
        "risk": "The AI system or its outputs are misused for deceptive or manipulative activity.",
        "timing": "pilot-operation",
        "pilot_relevance": "Define misuse cases, monitoring, access limits, and response procedures.",
        "mitigation": "Limit access, define prohibited uses, monitor anomalous activity, and prepare incident response steps.",
    },
    {
        "domain": "Human-Computer Interaction",
        "subdomain": "Overreliance and unsafe use",
        "risk": "Users trust or rely on AI outputs beyond the system's validated capability.",
        "timing": "pilot-operation",
        "pilot_relevance": "Design user guidance, confidence signals, and mandatory human checks.",
        "mitigation": "Set clear usage boundaries, train users, require human approval, and display uncertainty or source evidence.",
    },
    {
        "domain": "Human-Computer Interaction",
        "subdomain": "Loss of human agency and autonomy",
        "risk": "The pilot shifts meaningful control from responsible humans to the AI system.",
        "timing": "pilot-design",
        "pilot_relevance": "Keep decision rights, accountability, and override paths explicit.",
        "mitigation": "Define accountable roles, override procedures, appeal routes, and non-automated fallback processes.",
    },
    {
        "domain": "Socioeconomic & Environmental",
        "subdomain": "Governance failure",
        "risk": "Oversight, responsibilities, or controls are insufficient for the pilot context.",
        "timing": "pilot-design",
        "pilot_relevance": "Assign owners, review gates, acceptance criteria, and audit evidence.",
        "mitigation": "Create governance checkpoints, risk owners, documentation requirements, and pilot exit criteria.",
    },
    {
        "domain": "AI System Safety, Failures, & Limitations",
        "subdomain": "Lack of capability or robustness",
        "risk": "The AI system fails under realistic pilot conditions or edge cases.",
        "timing": "pilot-evaluation",
        "pilot_relevance": "Stress test with representative cases before operational use.",
        "mitigation": "Test on representative edge cases, define failure modes, and limit use to validated operating conditions.",
    },
    {
        "domain": "AI System Safety, Failures, & Limitations",
        "subdomain": "Lack of transparency or interpretability",
        "risk": "Users and reviewers cannot understand, contest, or correct AI outputs.",
        "timing": "pilot-operation",
        "pilot_relevance": "Define explanation, documentation, and traceability requirements.",
        "mitigation": "Document model behavior, preserve audit trails, expose sources, and define correction or appeal processes.",
    },
]


class FileIngestorGovTech:
    def __init__(self, uploaded_files=None, mode="Chat with reports"):
        self.uploaded_files = uploaded_files or []
        self.mode = mode

    def slugify(self, text):
        slug = os.path.splitext(text or "untitled")[0].lower()
        slug = re.sub(r"[^a-z0-9]+", "-", slug)
        slug = re.sub(r"-+", "-", slug).strip("-")
        return slug or "untitled"

    def extract_json_from_response(self, text):
        text = text.strip()
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            raise ValueError("No JSON object found in LLM response")

        return json.loads(text[start:end + 1])

    def infer_wiki_category(self, file_path):
        path_parts = set(file_path.lower().split(os.sep))

        for category in ["projects", "risks", "lessons", "regulations"]:
            if category in path_parts:
                return category

        return "general"

    def infer_external_source_type(self, file_path):
        path = file_path.lower()

        if "mit" in path or "ai_risk_repository" in path or "airisk" in path:
            return "mit_ai_risk_repository"

        if "nist" in path or "ai_rmf" in path or "risk_management_framework" in path:
            return "nist_ai_rmf"

        return "external_ai_risk_source"

    def first_matching_column(self, columns, patterns):
        normalized = {column.lower().strip(): column for column in columns}

        for pattern in patterns:
            for lower_column, original_column in normalized.items():
                if pattern in lower_column:
                    return original_column

        return None

    def normalize_mit_risk_records(self, rows, source):
        if not rows:
            return []

        columns = list(rows[0].keys())
        risk_col = self.first_matching_column(
            columns,
            ["risk", "description", "quote", "harm", "issue"],
        )
        domain_col = self.first_matching_column(
            columns,
            ["domain", "risk domain", "category"],
        )
        subdomain_col = self.first_matching_column(
            columns,
            ["subdomain", "sub-domain", "subcategory", "risk subdomain"],
        )
        timing_col = self.first_matching_column(
            columns,
            ["timing", "stage", "pre-deployment", "post-deployment"],
        )
        cause_col = self.first_matching_column(
            columns,
            ["cause", "causal", "entity", "intent"],
        )
        mitigation_col = self.first_matching_column(
            columns,
            ["mitigation", "control", "measure", "intervention", "recommendation"],
        )

        records = []

        for index, row in enumerate(rows):
            risk = str(row.get(risk_col, "")).strip() if risk_col else ""
            domain = str(row.get(domain_col, "")).strip() if domain_col else ""
            subdomain = (
                str(row.get(subdomain_col, "")).strip() if subdomain_col else ""
            )
            timing = str(row.get(timing_col, "")).strip() if timing_col else ""
            cause = str(row.get(cause_col, "")).strip() if cause_col else ""
            mitigation = (
                str(row.get(mitigation_col, "")).strip() if mitigation_col else ""
            )

            if not risk and not domain and not subdomain:
                continue

            if not risk:
                risk = subdomain or domain or f"MIT AI risk {index + 1}"

            if not domain:
                domain = "Unclassified"

            records.append(
                {
                    "risk_id": f"{self.slugify(source)}-{index + 1}",
                    "risk": risk[:600],
                    "domain": domain,
                    "subdomain": subdomain or domain,
                    "timing": timing or "unspecified",
                    "cause": cause,
                    "source": source,
                    "pilot_relevance": "",
                    "mitigation": mitigation,
                }
            )

        return records

    def load_mit_risk_records(self):
        records = []

        for source_dir in EXTERNAL_RISK_SOURCE_DIRS:
            if not os.path.isdir(source_dir):
                continue

            for root, _, files in os.walk(source_dir):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    lower_path = file_path.lower()

                    if not (
                        "mit" in lower_path
                        or "airisk" in lower_path
                        or "ai_risk_repository" in lower_path
                    ):
                        continue

                    _, extension = os.path.splitext(filename)
                    extension = extension.lower()

                    try:
                        if extension == ".csv":
                            dataframe = pd.read_csv(file_path)
                            rows = dataframe.fillna("").to_dict(orient="records")
                            records.extend(
                                self.normalize_mit_risk_records(rows, file_path)
                            )
                        elif extension == ".json":
                            with open(file_path, "r", encoding="utf-8") as json_file:
                                payload = json.load(json_file)

                            if isinstance(payload, dict):
                                for value in payload.values():
                                    if isinstance(value, list):
                                        payload = value
                                        break

                            if isinstance(payload, list):
                                records.extend(
                                    self.normalize_mit_risk_records(payload, file_path)
                                )
                    except Exception as e:
                        st.warning(f"Could not parse MIT risk data {file_path}: {e}")

        if records:
            return records

        return [
            {
                "risk_id": f"mit-taxonomy-{index + 1}",
                "risk": item["risk"],
                "domain": item["domain"],
                "subdomain": item["subdomain"],
                "timing": item["timing"],
                "cause": "MIT AI Risk Repository taxonomy",
                "source": "https://airisk.mit.edu/",
                "pilot_relevance": item["pilot_relevance"],
                "mitigation": item["mitigation"],
            }
            for index, item in enumerate(MIT_RISK_DOMAINS)
        ]

    def build_mit_risk_dataframe(self):
        records = self.load_mit_risk_records()
        dataframe = pd.DataFrame(records)

        if dataframe.empty:
            return dataframe

        domain_order = {
            domain: index + 1
            for index, domain in enumerate(sorted(dataframe["domain"].unique()))
        }
        timing_order = {
            "pilot-design": 1,
            "pre-deployment": 1,
            "predeployment": 1,
            "development": 1,
            "pilot-evaluation": 2,
            "evaluation": 2,
            "testing": 2,
            "pilot-operation": 3,
            "deployment": 3,
            "post-deployment": 3,
            "postdeployment": 3,
            "operation": 3,
            "unspecified": 2,
        }

        dataframe["domain_y"] = dataframe["domain"].map(domain_order)
        dataframe["timing_key"] = dataframe["timing"].astype(str).str.lower()
        dataframe["pilot_stage_x"] = dataframe["timing_key"].map(timing_order).fillna(2)
        dataframe["label"] = dataframe["risk"].astype(str).str.slice(0, 90)

        if "mitigation" not in dataframe.columns:
            dataframe["mitigation"] = ""

        dataframe["mitigation"] = dataframe["mitigation"].fillna("")
        dataframe["mitigation_card"] = dataframe["mitigation"].where(
            dataframe["mitigation"].str.strip() != "",
            "Generate the analysis report for recommended mitigations.",
        )

        return dataframe

    def generate_selected_risk_report(self, selected_risk, db, project_context):
        llm = Loadllm.load_llm()

        query = "\n".join(
            [
                selected_risk.get("risk", ""),
                selected_risk.get("domain", ""),
                selected_risk.get("subdomain", ""),
                selected_risk.get("pilot_relevance", ""),
                project_context,
                "AI pilot risk mitigation NIST AI RMF GovTech Sandbox",
            ]
        )

        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10},
        )
        docs = retriever.invoke(query)
        context = self.format_docs_context(docs)

        prompt = f"""
You are an AI pilot risk analyst.

Prepare a focused analysis report for the selected MIT AI Risk Repository point.

Use ONLY the selected risk and retrieved context.

Scope:
- Focus on risks during an AI pilot project.
- Do not analyze speculative catastrophic or downstream societal risks unless the
  selected risk and project context directly require it.
- Provide risk-management guidance, not legal advice.

Return:

## Selected Risk
## Why This Matters For The Pilot
## Likelihood And Impact
## Concrete Mitigation Measures
## Evidence From Sandbox Wiki
## Useful NIST-Style Controls
## Open Questions
## Sources Used

Selected risk:
{json.dumps(selected_risk, indent=2)}

Project context:
{project_context}

Retrieved context:
{context}
"""

        response = llm.invoke(prompt)
        return response.content

    def generate_overall_risk_assessment(self, dataframe, db, project_context):
        llm = Loadllm.load_llm()

        risk_sample = dataframe[
            ["risk", "domain", "subdomain", "timing", "mitigation_card"]
        ].head(30).to_dict(orient="records")

        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 12},
        )
        docs = retriever.invoke(
            "\n".join(
                [
                    project_context,
                    "overall AI pilot risk assessment mitigation NIST AI RMF Sandbox Wiki",
                ]
            )
        )
        context = self.format_docs_context(docs)

        prompt = f"""
You are an AI pilot risk manager.

Create an overall risk assessment for the described AI pilot using the MIT risk map,
the Sandbox Wiki context, and any available NIST-style mitigation context.

Use ONLY the provided context.

Scope rules:
- Focus on practical risks during the AI pilot.
- Do not focus on speculative catastrophic or downstream societal risks.
- Provide risk-management guidance, not legal advice.

Return:

## Overall Risk Assessment
State the overall risk level: Low / Medium / High, with a short justification.

## Top Risk Clusters
Use a table with columns:
- Cluster
- Why It Matters
- Pilot Stage
- Priority

## Recommended Controls
Use concrete mitigation measures.

## Immediate Next Steps Before Pilot Start

## Monitoring During The Pilot

## Residual Risks And Open Questions

## Sources Used

Project context:
{project_context}

MIT risk map sample:
{json.dumps(risk_sample, indent=2)}

Retrieved context:
{context}
"""

        response = llm.invoke(prompt)
        return response.content

    def is_risk_analysis_request(self, text):
        normalized_text = text.lower()
        risk_terms = [
            "risk analysis",
            "risk assessment",
            "assess risk",
            "assess risks",
            "analyze risk",
            "analyse risk",
            "analyze risks",
            "analyse risks",
            "risk management",
            "mitigation",
            "mitigate",
            "risk register",
        ]

        return any(term in normalized_text for term in risk_terms)

    def load_pdfs(self):
        all_documents = []

        for uploaded_file in self.uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            loader = PyPDFLoader(file_path=tmp_file_path)
            documents = loader.load()

            for doc in documents:
                doc.metadata["source"] = uploaded_file.name
                doc.metadata["type"] = "raw_pdf"

            all_documents.extend(documents)

        return all_documents

    def save_uploaded_pdfs_to_hub(self):
        if not os.path.isdir(GOVTECH_HUB_DIR):
            return {}

        os.makedirs(GOVTECH_HUB_REPORTS_DIR, exist_ok=True)
        source_paths = {}

        for uploaded_file in self.uploaded_files:
            source_slug = self.slugify(uploaded_file.name)
            destination = os.path.join(GOVTECH_HUB_REPORTS_DIR, f"{source_slug}.pdf")

            with open(destination, "wb") as output_file:
                output_file.write(uploaded_file.getvalue())

            source_paths[uploaded_file.name] = {
                "slug": source_slug,
                "path": destination,
                "wiki_path": f"../../context/reports/en/{source_slug}.pdf",
            }

        return source_paths

    def write_markdown_file(self, folder, slug, markdown):
        folder_path = os.path.join(GOVTECH_HUB_WIKI_DIR, folder)
        os.makedirs(folder_path, exist_ok=True)

        file_path = os.path.join(folder_path, f"{slug}.md")

        with open(file_path, "w", encoding="utf-8") as output_file:
            output_file.write(markdown.strip() + "\n")

        return file_path

    def markdown_value(self, value):
        if isinstance(value, list):
            if not value:
                return ""
            return "\n".join([f"- {item}" for item in value])

        return str(value or "")

    def build_generated_page_markdown(self, title, page_type, slug, fields, body_sections):
        today = date.today().isoformat()
        frontmatter_lines = [
            "---",
            f"title: {json.dumps(title)[1:-1]}",
            f"type: {page_type}",
            f"slug: {slug}",
        ]

        for key, value in fields.items():
            if isinstance(value, list):
                frontmatter_lines.append(
                    f"{key}: [{', '.join([str(item) for item in value])}]"
                )
            elif value not in [None, ""]:
                frontmatter_lines.append(f"{key}: {value}")

        frontmatter_lines.extend(
            [
                f"created: {today}",
                f"updated: {today}",
                "---",
                "",
                f"# {title}",
                "",
            ]
        )

        body = []

        for heading, value in body_sections:
            rendered_value = self.markdown_value(value)

            if not rendered_value:
                continue

            body.extend([f"## {heading}", "", rendered_value, ""])

        return "\n".join(frontmatter_lines + body)

    def generate_and_save_wiki_to_hub(self, raw_documents):
        if not os.path.isdir(GOVTECH_HUB_DIR):
            st.warning(
                "The cloned govtech-sandbox-hub repo was not found at "
                "external/govtech-sandbox-hub."
            )
            return []

        os.makedirs(GOVTECH_HUB_WIKI_DIR, exist_ok=True)
        source_paths = self.save_uploaded_pdfs_to_hub()

        combined_text_parts = []

        for doc in raw_documents:
            source = doc.metadata.get("source", "unknown_report")
            page = doc.metadata.get("page", "unknown")
            combined_text_parts.append(
                f"\n\nSOURCE: {source} | PAGE: {page}\n{doc.page_content}"
            )

        combined_text = "\n\n".join(combined_text_parts[:30])

        source_slugs = [
            source_info["slug"] for source_info in source_paths.values()
        ]
        source_slugs_json = json.dumps(source_slugs)

        prompt = f"""
You are updating the GovTech Sandbox Hub LLM Wiki from newly uploaded AI pilot documents.

Extract practical, source-grounded wiki entries.

Return ONLY raw valid JSON.

Do not add markdown fences, comments, or explanations.

Use this JSON structure:

{{
  "source_summary": {{
    "title": "...",
    "summary": "...",
    "publisher": "...",
    "year": "",
    "language": "en"
  }},
  "projects": [
    {{
      "title": "...",
      "summary": "...",
      "sector": "public-administration",
      "status": "completed",
      "key_learnings": ["..."],
      "risks": ["..."],
      "mitigations": ["..."]
    }}
  ],
  "risks": [
    {{
      "title": "...",
      "description": "...",
      "pilot_relevance": "...",
      "mitigation": "...",
      "related_projects": ["..."]
    }}
  ],
  "lessons": [
    {{
      "title": "...",
      "description": "...",
      "how_to_apply": ["..."],
      "confidence": "medium",
      "related_projects": ["..."]
    }}
  ],
  "regulations": [
    {{
      "title": "...",
      "description": "...",
      "jurisdiction": "ch-federal",
      "domain": ["data-protection"],
      "related_projects": ["..."]
    }}
  ]
}}

Rules:
- Use only information supported by the uploaded documents.
- Focus on practical AI pilot project risks and mitigations.
- Do not include speculative catastrophic AI risks unless the document itself discusses them.
- Keep titles short and suitable for wiki pages.
- If information is missing, use an empty string or empty list.
- Available source slugs for frontmatter references: {source_slugs_json}

Uploaded document text:
{combined_text}
"""

        generated_files = []

        try:
            llm = Loadllm.load_llm()
            response = llm.invoke(prompt)
            knowledge = self.extract_json_from_response(response.content)
        except Exception as e:
            st.warning(f"Could not generate wiki update JSON: {e}")
            return generated_files

        today = date.today().isoformat()

        for uploaded_name, source_info in source_paths.items():
            source_summary = knowledge.get("source_summary", {})
            title = source_summary.get("title") or os.path.splitext(uploaded_name)[0]
            source_slug = source_info["slug"]
            markdown = self.build_generated_page_markdown(
                title=title,
                page_type="source",
                slug=source_slug,
                fields={
                    "source_type": "pdf",
                    "path": source_info["wiki_path"],
                    "language": source_summary.get("language", "en"),
                    "year": source_summary.get("year", ""),
                    "publisher": source_summary.get("publisher", ""),
                },
                body_sections=[
                    ("Summary", source_summary.get("summary", "")),
                    ("Provenance", f"Original filename: `{uploaded_name}`."),
                    ("Use as citation", "Generated from a newly uploaded PDF."),
                ],
            )
            generated_files.append(
                self.write_markdown_file("sources", source_slug, markdown)
            )

        for item in knowledge.get("projects", []):
            title = item.get("title", "Untitled Project")
            slug = self.slugify(title)
            markdown = self.build_generated_page_markdown(
                title=title,
                page_type="project",
                slug=slug,
                fields={
                    "phase": "II",
                    "year": today[:4],
                    "status": item.get("status", "analysis-only"),
                    "sector": item.get("sector", "other"),
                    "sources": source_slugs,
                },
                body_sections=[
                    ("Summary", item.get("summary", "")),
                    ("Key Learnings", item.get("key_learnings", [])),
                    ("Risks", item.get("risks", [])),
                    ("Mitigations", item.get("mitigations", [])),
                ],
            )
            generated_files.append(
                self.write_markdown_file("projects", slug, markdown)
            )

        for item in knowledge.get("risks", []):
            title = item.get("title", "Untitled Risk")
            slug = f"risk-{self.slugify(title)}"
            markdown = self.build_generated_page_markdown(
                title=title,
                page_type="concept",
                slug=slug,
                fields={
                    "related": [],
                    "appears_in": source_slugs,
                },
                body_sections=[
                    ("Description", item.get("description", "")),
                    ("Pilot Relevance", item.get("pilot_relevance", "")),
                    ("Mitigation", item.get("mitigation", "")),
                    ("Related Projects", item.get("related_projects", [])),
                ],
            )
            generated_files.append(
                self.write_markdown_file("concepts", slug, markdown)
            )

        for item in knowledge.get("lessons", []):
            title = item.get("title", "Untitled Lesson")
            slug = self.slugify(title)
            markdown = self.build_generated_page_markdown(
                title=title,
                page_type="lesson",
                slug=slug,
                fields={
                    "phase": "II",
                    "project": [],
                    "concept": [],
                    "regulation": [],
                    "sources": source_slugs,
                    "confidence": item.get("confidence", "medium"),
                    "freshness": today[:7],
                    "applies_to_lifecycle_stage": ["scoping", "pilot"],
                    "cross_cutting": "false",
                },
                body_sections=[
                    ("Description", item.get("description", "")),
                    ("How To Apply", item.get("how_to_apply", [])),
                    ("Related Projects", item.get("related_projects", [])),
                ],
            )
            generated_files.append(
                self.write_markdown_file("lessons", slug, markdown)
            )

        for item in knowledge.get("regulations", []):
            title = item.get("title", "Untitled Regulation")
            slug = self.slugify(title)
            markdown = self.build_generated_page_markdown(
                title=title,
                page_type="regulation",
                slug=slug,
                fields={
                    "jurisdiction": item.get("jurisdiction", "other"),
                    "instrument": "other",
                    "domain": item.get("domain", []),
                    "year": today[:4],
                    "project": [],
                },
                body_sections=[
                    ("Description", item.get("description", "")),
                    ("Related Projects", item.get("related_projects", [])),
                    ("Sources", source_slugs),
                ],
            )
            generated_files.append(
                self.write_markdown_file("regulations", slug, markdown)
            )

        return generated_files

    def load_existing_wiki_pages(self):
        wiki_documents = []
        seen_files = set()

        for source_dir in WIKI_SOURCE_DIRS:
            if not os.path.isdir(source_dir):
                continue

            for root, _, files in os.walk(source_dir):
                for filename in files:
                    _, extension = os.path.splitext(filename)

                    if extension.lower() not in WIKI_FILE_EXTENSIONS:
                        continue

                    file_path = os.path.join(root, filename)
                    absolute_path = os.path.abspath(file_path)

                    if absolute_path in seen_files:
                        continue

                    seen_files.add(absolute_path)

                    try:
                        with open(file_path, "r", encoding="utf-8") as wiki_file:
                            content = wiki_file.read().strip()
                    except UnicodeDecodeError:
                        with open(file_path, "r", encoding="latin-1") as wiki_file:
                            content = wiki_file.read().strip()
                    except Exception as e:
                        st.warning(f"Could not load wiki page {file_path}: {e}")
                        continue

                    if not content:
                        continue

                    category = self.infer_wiki_category(file_path)

                    wiki_documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": file_path,
                                "type": f"existing_llm_wiki_{category}",
                                "page": "wiki_page",
                                "category": category,
                            },
                        )
                    )

        return wiki_documents

    def load_external_risk_sources(self):
        external_documents = []
        seen_files = set()

        for source_dir in EXTERNAL_RISK_SOURCE_DIRS:
            if not os.path.isdir(source_dir):
                continue

            for root, _, files in os.walk(source_dir):
                for filename in files:
                    _, extension = os.path.splitext(filename)
                    extension = extension.lower()

                    if extension not in EXTERNAL_RISK_FILE_EXTENSIONS:
                        continue

                    file_path = os.path.join(root, filename)
                    absolute_path = os.path.abspath(file_path)

                    if absolute_path in seen_files:
                        continue

                    seen_files.add(absolute_path)
                    source_type = self.infer_external_source_type(file_path)

                    if extension == ".pdf":
                        try:
                            loader = PyPDFLoader(file_path=file_path)
                            documents = loader.load()
                        except Exception as e:
                            st.warning(f"Could not load external PDF {file_path}: {e}")
                            continue

                        for doc in documents:
                            doc.metadata["source"] = file_path
                            doc.metadata["type"] = source_type

                        external_documents.extend(documents)
                        continue

                    try:
                        with open(file_path, "r", encoding="utf-8") as source_file:
                            content = source_file.read().strip()
                    except UnicodeDecodeError:
                        with open(file_path, "r", encoding="latin-1") as source_file:
                            content = source_file.read().strip()
                    except Exception as e:
                        st.warning(f"Could not load external source {file_path}: {e}")
                        continue

                    if not content:
                        continue

                    external_documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": file_path,
                                "type": source_type,
                                "page": "external_source",
                            },
                        )
                    )

        return external_documents

    def build_vector_db(self, documents):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
        )

        chunks = splitter.split_documents(documents)
        chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

        if not chunks:
            st.error("No valid text chunks found.")
            return None

        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(DB_FAISS_PATH)

        return db

    def format_docs_context(self, docs):
        return "\n\n".join(
            [
                f"[Source: {doc.metadata.get('source', 'Unknown')} | "
                f"Type: {doc.metadata.get('type', 'Unknown')} | "
                f"Page: {doc.metadata.get('page', 'Unknown')}]\n"
                f"{doc.page_content}"
                for doc in docs
            ]
        )

    def run_readiness_advisor(self, db):
        st.subheader("AI Project Readiness Advisor")

        project_description = st.text_area(
            "Describe your planned AI project",
            placeholder="Example: We want to use AI to process citizen applications...",
        )

        if st.button("Analyze project") and project_description:
            llm = Loadllm.load_llm()

            retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 8},
            )

            docs = retriever.invoke(project_description)
            context = self.format_docs_context(docs)

            prompt = f"""
You are an AI Project Readiness Advisor for Swiss public-sector AI projects.

Use ONLY the provided context from existing LLM wiki pages and uploaded reports.

Return:

## Similar Sandbox Projects
## Relevant Lessons Learned
## Legal Risks
## Data Protection Risks
## Technical Risks
## Organizational Risks
## Recommended Next Steps

## AI Project Readiness Checklist
Use checkbox format:
- [ ] ...

## Sources Used

If the context is insufficient, clearly say what is missing.

Project description:
{project_description}

Context:
{context}
"""

            response = llm.invoke(prompt)
            st.markdown(response.content)

    def run_risk_dashboard(self, wiki_documents):
        st.subheader("AI Risk Dashboard")

        if not wiki_documents:
            st.warning("No existing LLM wiki pages available.")
            return

        llm = Loadllm.load_llm()

        wiki_context = "\n\n".join(
            [
                f"[Source: {doc.metadata.get('source', 'Unknown')} | "
                f"Category: {doc.metadata.get('category', 'Unknown')}]\n"
                f"{doc.page_content}"
                for doc in wiki_documents
            ]
        )

        prompt = f"""
Analyze these existing AI Innovation Sandbox LLM wiki pages.

Use ONLY the provided wiki pages.

Create a risk dashboard with:

## Risk Overview Table
Columns:
- Risk Category
- Risk Description
- Affected Projects
- Severity: Low / Medium / High
- Recommended Mitigation

## Top Recurring Risks
## Most Common Legal Risks
## Most Common Technical Risks
## Most Common Organizational Risks
## Common Governance Themes
## Priority Actions For Future AI Projects

Wiki pages:
{wiki_context}
"""

        response = llm.invoke(prompt)
        st.markdown(response.content)

    def run_ai_pilot_risk_agent(self, db):
        st.subheader("AI Pilot Risk Management Agent")

        project_description = st.text_area(
            "Describe the planned AI pilot",
            placeholder=(
                "Example: A municipality wants to pilot an AI assistant that helps "
                "case workers summarize citizen applications containing personal data."
            ),
            height=180,
        )

        sector = st.text_input(
            "Sector or organization type",
            placeholder="Example: public administration, education, healthcare, finance",
        )

        data_context = st.text_area(
            "Data, users, and deployment context",
            placeholder=(
                "Example: Uses internal documents and citizen records; human case "
                "workers review all outputs; pilot duration is 3 months."
            ),
            height=120,
        )

        if st.button("Assess AI pilot risks") and project_description:
            llm = Loadllm.load_llm()

            retrieval_query = "\n".join(
                [
                    project_description,
                    sector,
                    data_context,
                    "AI pilot project risks mitigation public sector regulated sector",
                    "MIT AI Risk Repository NIST AI RMF governance map measure manage",
                ]
            )

            retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 12},
            )

            docs = retriever.invoke(retrieval_query)
            context = self.format_docs_context(docs)

            prompt = f"""
You are an AI Pilot Risk Management Agent for employees in highly regulated sectors.

Your task is to identify, evaluate, and mitigate risks that can occur DURING an AI pilot
project in an organization.

Use ONLY the provided context from:
- GovTech Sandbox LLM Wiki pages based on practical pilot experience
- MIT AI Risk Repository material, when available
- NIST AI Risk Management Framework material, when available
- Uploaded project documents, when available

Very important scope rules:
- Focus on practical pilot-project risks: legal, data protection, security, technical,
  operational, procurement, organizational, human oversight, evaluation, transparency,
  bias, accuracy, accountability, vendor, and stakeholder risks.
- Do NOT focus on speculative downstream or catastrophic AI risks such as extinction,
  large-scale macroeconomic disruption, superintelligence, or geopolitical race dynamics
  unless the provided project description directly requires it.
- Do NOT give legal advice. Provide risk-management guidance and note when legal review
  is needed.
- If the evidence is insufficient, say what information is missing.
- Distinguish practical Sandbox experience from broader external frameworks where useful.

Return the answer in this structure:

## Executive Summary

## Pilot Context Assumptions

## Risk Register
Use a Markdown table with columns:
- Risk
- Why It Matters In This Pilot
- Likelihood: Low / Medium / High
- Impact: Low / Medium / High
- Priority: Low / Medium / High
- Evidence Source

## Mitigation Measures
Map each key risk to concrete mitigation actions. Prefer NIST-style governance,
mapping, measurement, and management actions when supported by context.

## Pilot Go / No-Go Considerations

## Questions To Ask Before Starting The Pilot

## Sources Used

Project description:
{project_description}

Sector or organization type:
{sector}

Data, users, and deployment context:
{data_context}

Retrieved context:
{context}
"""

            response = llm.invoke(prompt)
            st.markdown(response.content)

        st.divider()
        self.run_mit_risk_analysis_plot(db, project_description, sector, data_context)

    def run_mit_risk_analysis_plot(self, db, project_description="", sector="", data_context=""):
        st.subheader("Risk Map")

        dataframe = self.build_mit_risk_dataframe()

        if dataframe.empty:
            st.warning("No MIT risk records available.")
            return

        selected_risk = None

        try:
            import plotly.express as px
            from streamlit_plotly_events import plotly_events

            figure = px.scatter(
                dataframe,
                x="pilot_stage_x",
                y="domain_y",
                color="domain",
                hover_name="label",
                hover_data={
                    "risk": True,
                    "subdomain": True,
                    "timing": True,
                    "mitigation_card": True,
                    "source": True,
                    "pilot_stage_x": False,
                    "domain_y": False,
                },
                custom_data=[
                    "risk_id",
                    "domain",
                    "subdomain",
                    "timing",
                    "mitigation_card",
                ],
                labels={
                    "pilot_stage_x": "Pilot stage",
                    "domain_y": "MIT risk domain",
                    "domain": "Domain",
                    "mitigation_card": "Mitigation steps",
                },
            )
            figure.update_traces(
                marker={
                    "size": 11,
                    "opacity": 0.82,
                    "line": {"width": 1, "color": "white"},
                },
                hovertemplate=(
                    "<b>%{hovertext}</b><br>"
                    "Domain: %{customdata[1]}<br>"
                    "Subdomain: %{customdata[2]}<br>"
                    "Stage: %{customdata[3]}<br><br>"
                    "<b>Mitigation card</b><br>%{customdata[4]}"
                    "<extra></extra>"
                ),
            )
            figure.update_xaxes(
                tickmode="array",
                tickvals=[1, 2, 3],
                ticktext=["Design", "Evaluation", "Operation"],
            )
            figure.update_yaxes(
                tickmode="array",
                tickvals=sorted(dataframe["domain_y"].unique()),
                ticktext=[
                    domain
                    for domain, _ in sorted(
                        {
                            row["domain"]: row["domain_y"]
                            for _, row in dataframe.iterrows()
                        }.items(),
                        key=lambda item: item[1],
                    )
                ],
            )
            figure.update_layout(height=620, margin={"l": 40, "r": 20, "t": 20, "b": 40})

            clicked_points = plotly_events(
                figure,
                click_event=True,
                hover_event=False,
                select_event=False,
                override_height=620,
                key="mit_risk_plot",
            )

            if clicked_points:
                point_number = clicked_points[0].get("pointNumber")
                curve_number = clicked_points[0].get("curveNumber")
                clicked_trace = figure.data[curve_number]
                risk_id = clicked_trace.customdata[point_number][0]
                selected_rows = dataframe[dataframe["risk_id"] == risk_id]

                if not selected_rows.empty:
                    selected_risk = selected_rows.iloc[0].to_dict()
                    st.session_state["last_selected_risk_id"] = risk_id

        except Exception:
            chart = (
                alt.Chart(dataframe)
                .mark_circle(size=95, opacity=0.82, stroke="white", strokeWidth=1)
                .encode(
                    x=alt.X(
                        "pilot_stage_x:Q",
                        title="Pilot stage: 1 design, 2 evaluation, 3 operation",
                        scale=alt.Scale(domain=[0.5, 3.5]),
                    ),
                    y=alt.Y("domain:N", title="MIT risk domain"),
                    color=alt.Color("domain:N", legend=None),
                    tooltip=[
                        alt.Tooltip("risk:N", title="Risk"),
                        alt.Tooltip("domain:N", title="Domain"),
                        alt.Tooltip("subdomain:N", title="Subdomain"),
                        alt.Tooltip("timing:N", title="Pilot stage"),
                        alt.Tooltip("mitigation_card:N", title="Mitigation steps"),
                        alt.Tooltip("source:N", title="Source"),
                    ],
                )
                .properties(height=520)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

        if selected_risk is None:
            selected_rows = dataframe.head(0)

            if "last_selected_risk_id" in st.session_state:
                selected_rows = dataframe[
                    dataframe["risk_id"] == st.session_state["last_selected_risk_id"]
                ]

            if not selected_rows.empty:
                selected_risk = selected_rows.iloc[0].to_dict()

        if not selected_risk:
            return

        st.session_state["last_selected_risk_id"] = selected_risk.get("risk_id", "")

        st.markdown(
            f"""
<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px 16px;margin-top:10px;background:#ffffff;">
  <div style="font-size:13px;color:#57606a;margin-bottom:4px;">{selected_risk.get('domain', '')}</div>
  <div style="font-weight:700;font-size:16px;margin-bottom:8px;">{selected_risk.get('subdomain', '')}</div>
  <div style="margin-bottom:10px;">{selected_risk.get('risk', '')}</div>
  <div style="font-size:13px;color:#57606a;margin-bottom:4px;">Mitigation</div>
  <div>{selected_risk.get('mitigation_card', '')}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    def run_chat(self, db):
        if "history" not in st.session_state:
            st.session_state["history"] = []

        if "risk_analysis_query" not in st.session_state:
            st.session_state["risk_analysis_query"] = ""

        if "generated" not in st.session_state:
            st.session_state["generated"] = [
                "Hello! Ask me about the GovTech LLM wiki or uploaded reports."
            ]

        if "past" not in st.session_state:
            st.session_state["past"] = ["Hey! GovTech Assistant"]

        chat_history_container = st.container()
        live_stream_container = st.container()
        input_container = st.container()

        with chat_history_container:
            for i in range(len(st.session_state["generated"])):
                message(
                    st.session_state["past"][i],
                    is_user=True,
                    key=f"{i}_user",
                    avatar_style="initials",
                    seed="User",
                )

                message(
                    st.session_state["generated"][i],
                    key=f"{i}",
                    avatar_style="initials",
                    seed="GovTech Assistant",
                )

        with live_stream_container:
            loading_placeholder = st.empty()
            chat_bot_message_placeholder = st.empty()

        with input_container:
            with st.form(key="my_form", clear_on_submit=True):
                user_input = st.text_input(
                    "Ask a question",
                    placeholder="Ask about AI sandbox reports...",
                    key="input",
                    label_visibility="collapsed",
                )
                submit_button = st.form_submit_button(label="Send")

        if submit_button and user_input:
            st.session_state["past"].append(user_input)

            with chat_history_container:
                message(
                    user_input,
                    is_user=True,
                    key=f"{len(st.session_state['past'])}_user",
                    avatar_style="initials",
                    seed="User",
                )

            with live_stream_container:
                with loading_placeholder.container():
                    with st.spinner("Thinking ..."):
                        llm = Loadllm.load_llm()

                        retriever = db.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 7},
                        )

                        docs = retriever.invoke(user_input)
                        context = self.format_docs_context(docs)

                        prompt = f"""
You are a trustworthy GovTech AI knowledge assistant.

You answer questions about existing LLM wiki pages and uploaded GovTech reports.

Use ONLY the provided context.

Important rules:
- If the answer is not supported by the context, say:
  "I don't know based on the available wiki pages and uploaded documents."
- Prefer practical, clear answers for non-technical public-sector users.
- Mention source file names when useful.
- Distinguish between:
  1. evidence from uploaded PDF chunks
  2. existing LLM wiki pages
- Do not invent legal, technical, or organizational claims.

Context:
{context}

User question:
{user_input}

Answer:
"""

                        response = llm.invoke(prompt)
                        answer = response.content

                loading_placeholder.empty()
                chat_bot_message_placeholder.markdown(answer)

            st.session_state["generated"].append(answer)
            st.session_state["history"].append((user_input, answer))

            if self.is_risk_analysis_request(user_input):
                st.session_state["risk_analysis_query"] = user_input

        if st.session_state.get("risk_analysis_query"):
            st.divider()
            self.run_mit_risk_analysis_plot(
                db,
                project_description=st.session_state["risk_analysis_query"],
            )

    def handlefileandingest(self):
        raw_documents = self.load_pdfs()

        if raw_documents and os.path.isdir(GOVTECH_HUB_DIR):
            st.info(
                "Uploaded PDFs can be added to the local govtech-sandbox-hub wiki. "
                "This writes Markdown files into external/govtech-sandbox-hub/wiki."
            )

            if st.button("Update cloned LLM Wiki from uploaded PDFs"):
                with st.spinner("Generating wiki pages in the cloned repository..."):
                    generated_files = self.generate_and_save_wiki_to_hub(raw_documents)

                if generated_files:
                    st.success(
                        f"Updated cloned wiki with {len(generated_files)} file(s). "
                        "Review, commit, and push them from external/govtech-sandbox-hub."
                    )

                    with st.expander("Generated wiki files"):
                        for file_path in generated_files:
                            st.code(file_path)
                else:
                    st.warning("No wiki files were generated.")

        wiki_documents = self.load_existing_wiki_pages()
        external_risk_documents = self.load_external_risk_sources()
        all_documents = raw_documents + wiki_documents + external_risk_documents

        if not all_documents:
            st.error(
                "No content found. Add existing LLM wiki Markdown files under "
                "wiki_pages/, clone govtech-sandbox-hub under external/, or add "
                "MIT/NIST source files under external/risk_sources/."
            )
            return

        with st.spinner("Building FAISS vector database from available sources..."):
            db = self.build_vector_db(all_documents)

        if db is None:
            return

        st.success(
            f"Loaded {len(wiki_documents)} existing LLM wiki page(s) and "
            f"{len(external_risk_documents)} external risk source document(s) and "
            f"{len(self.uploaded_files)} uploaded PDF report(s)."
        )

        if self.mode == "AI Pilot Risk Management Agent":
            self.run_ai_pilot_risk_agent(db)
            return

        if self.mode == "AI Project Readiness Advisor":
            self.run_readiness_advisor(db)
            return

        if self.mode == "Risk Dashboard":
            self.run_risk_dashboard(wiki_documents)
            return

        self.run_chat(db)
