import os
import re
import tempfile
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


class FileIngestorGovTechWikiPerPage:
    def __init__(self, uploaded_files, mode="Chat with reports"):
        self.uploaded_files = uploaded_files
        self.mode = mode

    def safe_filename(self, filename):
        name = os.path.splitext(filename)[0]
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        return name

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

    def generate_llm_wiki_pages(self, raw_documents):
        os.makedirs(WIKI_DIR, exist_ok=True)

        llm = Loadllm.load_llm()
        grouped_by_source = {}

        for doc in raw_documents:
            source = doc.metadata.get("source", "unknown_report")
            grouped_by_source.setdefault(source, [])
            grouped_by_source[source].append(doc.page_content)

        wiki_documents = []

        for source, pages in grouped_by_source.items():
            report_text = "\n\n".join(pages[:8])

            prompt = f"""
You are creating a lightweight LLM Wiki page for a GovTech AI Innovation Sandbox report.

Transform the report text into a structured reusable knowledge page.

Extract only information supported by the report text.

Return markdown with these sections:

# Report Summary
# AI Use Case
# Sector / Domain
# Data Used
# Legal or Regulatory Issues
# Technical Challenges
# Organizational Challenges
# Lessons Learned
# Recommendations
# Related Themes

Report source:
{source}

Report text:
{report_text}
"""

            try:
                response = llm.invoke(prompt)
                wiki_text = response.content

                safe_name = self.safe_filename(source)
                wiki_file_path = os.path.join(WIKI_DIR, f"{safe_name}.md")

                with open(wiki_file_path, "w", encoding="utf-8") as f:
                    f.write(wiki_text)

                with st.expander(f"📘 LLM Wiki Page: {source}"):
                    st.markdown(wiki_text)

                wiki_doc = Document(
                    page_content=wiki_text,
                    metadata={
                        "source": source,
                        "type": "llm_wiki",
                        "page": "wiki_summary",
                        "wiki_file": wiki_file_path
                    }
                )

                wiki_documents.append(wiki_doc)

            except Exception as e:
                st.warning(f"Could not generate LLM Wiki page for {source}: {e}")

        return wiki_documents

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