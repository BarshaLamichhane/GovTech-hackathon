# Screening Call Preparation

This guide is for the Azure/RAG screening call using the existing GovTech AI project.

## What to show

1. Run the demo app:
   - `streamlit run app-azure-screening-demo.py`
   - Show the upload and retrieval workflow.
   - Highlight how the app ingests PDF reports and local wiki content, builds a RAG index, and answers questions.

2. Use the existing `app-azure-screening-demo.py` as a live prototype.
   - It is a copy of the current GovTech app with a screening-call title and notes.
   - Explain that this prototype is already architected for RAG and can map to Azure.

## Key topics to discuss

- **Azure AI-102 certification**
  - Mention familiarity with Azure AI fundamentals, responsible AI, and deploying AI solutions on Azure.
  - Emphasize knowledge of Azure AI services, security, and enterprise governance.

- **Azure AI Search**
  - Describe it as the retrieval layer for RAG.
  - Explain how PDFs and documentation are indexed into Azure Cognitive Search and then combined with embeddings for retrieval.

- **Document Intelligence**
  - Talk about using Document Intelligence to extract structured content from PDFs, forms, and reports.
  - Say that the current app ingests PDFs and can be extended to Azure Document Intelligence for stronger extraction and metadata handling.

- **Prompt Flow**
  - Say you would use Prompt Flow to manage prompts, test prompt variations, and orchestrate multi-step workflows.
  - Describe how a prompt flow can supervise retrieval, summarization, and compliance checks.

- **RAG projects**
  - Point to this app as a RAG prototype: local FAISS vector store, embeddings, and chat/retrieval.
  - Discuss how the same pattern translates to Azure: Azure AI Search + embeddings + LLM response generation.

- **Productionization**
  - Outline the path from prototype to production:
    - define data ingestion and indexing pipeline
    - add enterprise security, RBAC, identity, encryption in transit/at rest
    - add monitoring, logging, and model-version controls
    - add prompt governance and prompt testing
    - deploy via Azure App Service or containerized service

## Demo talking points

- "This app is currently built as a local LangChain + FAISS RAG prototype."
- "For Azure, I would swap the retrieval backend to Azure Cognitive Search and use Document Intelligence for PDF extraction."
- "I would use Prompt Flow to orchestrate the prompt chain and make the workflow observable and repeatable."
- "This project already shows the key RAG elements: ingest, embed, index, retrieve, and answer."

## Practical preparation

- Open the app and test one or two PDFs before the call.
- Prepare one example question that shows the system retrieving relevant context.
- Be ready to explain the architecture clearly in 3–4 sentences.
- Keep the conversation focused on enterprise readiness and migration from prototype to Azure production.

## Files to reference

- `app-azure-screening-demo.py`
- `app-govtech.py`
- `fileingestorGovTech.py`
- `README.md`

## Notes

This new demo file is intentionally a copy of the existing GovTech app. It makes the same functionality available under a screening-demo name and keeps the core prototype unchanged.
