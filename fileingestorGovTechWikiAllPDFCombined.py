import os
import re
import tempfile
import json
import streamlit as st
from streamlit_chat import message

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from loadllm import Loadllm


DB_FAISS_PATH = "vectorstore/db_faiss"
WIKI_DIR = "wiki_pages"


class FileIngestorGovTech:
    def __init__(self, uploaded_files, mode="Chat with reports"):
        self.uploaded_files = uploaded_files
        self.mode = mode

    def safe_filename(self, filename):
        name = os.path.splitext(filename)[0]
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        return name
    
    def extract_json_from_response(self, text):

        text = text.strip()

        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            raise ValueError("No JSON object found in LLM response")

        json_text = text[start:end + 1]

        return json.loads(json_text)

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

    ##################################################################################################
    def generate_llm_wiki_pages(self, raw_documents):
        os.makedirs(WIKI_DIR, exist_ok=True)

        for folder in ["projects", "risks", "lessons", "regulations"]:
            os.makedirs(os.path.join(WIKI_DIR, folder), exist_ok=True)

        llm = Loadllm.load_llm()

        combined_text_parts = []

        for doc in raw_documents:
            source = doc.metadata.get("source", "unknown_report")
            page = doc.metadata.get("page", "unknown")

            combined_text_parts.append(
                f"\n\nSOURCE: {source} | PAGE: {page}\n{doc.page_content}"
            )

        combined_text = "\n\n".join(combined_text_parts[:20])

        prompt = f"""
You are building an LLM Wiki for AI Innovation Sandbox reports.

Extract structured knowledge from the provided report text.

Return ONLY raw valid JSON.

DO NOT:
- add explanations
- add markdown
- add comments
- add ```json fences
- write text before or after the JSON

The FIRST character of your response must be {{
The LAST character of your response must be }}

Return JSON with this structure:

{{
  "projects": [
    {{
      "title": "...",
      "summary": "...",
      "ai_use_case": "...",
      "sector": "...",
      "data_used": "...",
      "risks": ["..."],
      "lessons": ["..."],
      "regulations": ["..."],
      "sources": ["..."]
    }}
  ],

  "risks": [
    {{
      "title": "...",
      "description": "...",
      "affected_projects": ["..."],
      "mitigation": "...",
      "sources": ["..."]
    }}
  ],

  "lessons": [
    {{
      "title": "...",
      "description": "...",
      "related_projects": ["..."],
      "sources": ["..."]
    }}
  ],

  "regulations": [
    {{
      "title": "...",
      "description": "...",
      "related_risks": ["..."],
      "sources": ["..."]
    }}
  ]
}}

Important rules:
- Use ONLY information supported by the report text.
- Do NOT invent unsupported facts.
- Keep titles short and clear.
- Sources should mention PDF file names when possible.
- If information is missing, use an empty list [] or empty string "".
- Ensure the JSON is syntactically valid.

Report text:
{combined_text}
"""

        wiki_documents = []

        try:
            response = llm.invoke(prompt)
            raw_json = response.content.strip()

            st.code(raw_json[:3000], language="json")

            knowledge = self.extract_json_from_response(raw_json)

        except Exception as e:
            st.warning(f"Could not generate structured LLM Wiki JSON: {e}")
            return wiki_documents

        def save_wiki_page(category, item):
            title = item.get("title", "untitled")
            safe_name = self.safe_filename(title)

            folder_path = os.path.join(WIKI_DIR, category)
            file_path = os.path.join(folder_path, f"{safe_name}.md")

            markdown = f"# {title}\n\n"

            for key, value in item.items():
                if key == "title":
                    continue

                section_title = key.replace("_", " ").title()
                markdown += f"## {section_title}\n\n"

                if isinstance(value, list):
                    for v in value:
                        markdown += f"- {v}\n"
                    markdown += "\n"
                else:
                    markdown += f"{value}\n\n"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(markdown)

            with st.expander(f"📘 {category.title()} Wiki Page: {title}"):
                st.markdown(markdown)

            return Document(
                page_content=markdown,
                metadata={
                    "source": file_path,
                    "type": f"llm_wiki_{category}",
                    "page": "wiki_page",
                    "category": category
                }
            )

        for category in ["projects", "risks", "lessons", "regulations"]:
            for item in knowledge.get(category, []):
                wiki_documents.append(save_wiki_page(category, item))

        return wiki_documents
    ##################################################################################################
     

            

    def build_vector_db(self, documents):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100
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
        st.subheader("🧭 AI Project Readiness Advisor")

        project_description = st.text_area(
            "Describe your planned AI project",
            placeholder="Example: We want to use AI to process citizen applications..."
        )

        if st.button("Analyze project") and project_description:
            llm = Loadllm.load_llm()

            retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 8}
            )

            docs = retriever.invoke(project_description)
            context = self.format_docs_context(docs)

            prompt = f"""
You are an AI Project Readiness Advisor for Swiss public-sector AI projects.

Use ONLY the provided context from previous AI Innovation Sandbox reports.

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
        st.subheader("📊 AI Risk Dashboard")

        if not wiki_documents:
            st.warning("No LLM Wiki pages available.")
            return

        llm = Loadllm.load_llm()

        wiki_context = "\n\n".join(
            [
                f"[Source: {doc.metadata.get('source', 'Unknown')}]\n"
                f"{doc.page_content}"
                for doc in wiki_documents
            ]
        )

        prompt = f"""
Analyze these AI Innovation Sandbox wiki summaries.

Use ONLY the provided wiki summaries.

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

Wiki summaries:
{wiki_context}
"""

        response = llm.invoke(prompt)
        st.markdown(response.content)

    def run_chat(self, db):
        if "history" not in st.session_state:
            st.session_state["history"] = []

        if "generated" not in st.session_state:
            st.session_state["generated"] = [
                "Hello! Ask me about the uploaded GovTech reports."
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
                    seed="User"
                )

                message(
                    st.session_state["generated"][i],
                    key=f"{i}",
                    avatar_style="initials",
                    seed="GovTech Assistant"
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
                    label_visibility="collapsed"
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
                    seed="User"
                )

            with live_stream_container:
                with loading_placeholder.container():
                    with st.spinner("🧠 Thinking ..."):
                        llm = Loadllm.load_llm()

                        retriever = db.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 7}
                        )

                        docs = retriever.invoke(user_input)
                        context = self.format_docs_context(docs)

                        prompt = f"""
You are a trustworthy GovTech AI knowledge assistant.

You answer questions about AI Innovation Sandbox reports.

Use ONLY the provided context.

Important rules:
- If the answer is not supported by the context, say:
  "I don't know based on the uploaded documents."
- Prefer practical, clear answers for non-technical public-sector users.
- Mention source file names when useful.
- Distinguish between:
  1. evidence from original PDF chunks
  2. synthesized LLM Wiki summaries
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

    def handlefileandingest(self):
        raw_documents = self.load_pdfs()

        if not raw_documents or not any(doc.page_content.strip() for doc in raw_documents):
            st.error("No readable text found in the uploaded PDFs.")
            return

        with st.spinner("Creating lightweight LLM Wiki pages..."):
            wiki_documents = self.generate_llm_wiki_pages(raw_documents)

        all_documents = raw_documents + wiki_documents

        with st.spinner("Building FAISS vector database..."):
            db = self.build_vector_db(all_documents)

        if db is None:
            return

        st.success(
            f"Ingested {len(self.uploaded_files)} PDF(s) with "
            f"{len(wiki_documents)} LLM Wiki page(s)."
        )

        if self.mode == "AI Project Readiness Advisor":
            self.run_readiness_advisor(db)
            return

        if self.mode == "Risk Dashboard":
            self.run_risk_dashboard(wiki_documents)
            return

        self.run_chat(db)