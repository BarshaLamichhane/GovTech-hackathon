import streamlit as st

from fileingestorGovTech import FileIngestorGovTech
from streamlit_extras.colored_header import colored_header

st.title(":blue[Azure RAG Screening Demo — GovTech Knowledge Bot]")

st.markdown(
    "This demo is a copy of the existing GovTech knowledge assistant prepared for a screening call focused on Azure AI, RAG, and enterprise readiness."
)

colored_header(label="", description="", color_name="gray-30")

logo_link = "Hackathon ENG.png"
st.sidebar.image(logo_link)

st.sidebar.markdown(
    "### Screening call demo\n"
    "- Talk about Azure AI Search, Document Intelligence, Prompt Flow, and RAG.\n"
    "- Show how the same ingestion + retrieval flow can map to Azure enterprise integration.\n"
    "- Use this app to demonstrate a working prototype and discuss productionization."
)

mode = st.sidebar.radio(
    "Choose mode",
    [
        "Chat with reports",
        "AI Project Readiness Advisor",
        "Risk Dashboard"
    ]
)

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF reports (optional)",
    type="pdf",
    accept_multiple_files=True
)

file_ingestor = FileIngestorGovTech(uploaded_files, mode)
file_ingestor.handlefileandingest()
