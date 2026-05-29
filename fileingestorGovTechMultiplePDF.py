import os
import tempfile
import streamlit as st
from streamlit_chat import message

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from loadllm import Loadllm

DB_FAISS_PATH = "vectorstore/db_faiss"


class FileIngestorGovTech:
    def __init__(self, uploaded_files):
        self.uploaded_files = uploaded_files

    def handlefileandingest(self):

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        all_documents = []

        # Load all uploaded PDFs
        for uploaded_file in self.uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
                print("tmp_file_path:", tmp_file_path)

            loader = PyPDFLoader(file_path=tmp_file_path)
            documents = loader.load()

            for doc in documents:
                doc.metadata["source"] = uploaded_file.name

            all_documents.extend(documents)

        if not all_documents or not any(doc.page_content.strip() for doc in all_documents):
            st.error("No readable text found in the uploaded PDFs.")
            return

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(all_documents)
        chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

        if not chunks:
            st.error("No valid text chunks found in the uploaded PDFs.")
            return

        # Build one combined FAISS index
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(DB_FAISS_PATH)

        st.success(f"Ingested {len(self.uploaded_files)} PDF(s) into one FAISS index.")

        # Initialize session state
        if "history" not in st.session_state:
            st.session_state["history"] = []

        if "generated" not in st.session_state:
            st.session_state["generated"] = [
                "Hello! Ask me about the uploaded GovTech reports."
            ]

        if "past" not in st.session_state:
            st.session_state["past"] = ["Hey! iDOC AI"]

        chat_history_container = st.container()
        live_stream_container = st.container()
        input_container = st.container()

        # Display chat history
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
                    seed="Brikshyan Solutions"
                )

        with live_stream_container:
            loading_placeholder = st.empty()
            chat_bot_message_placeholder = st.empty()

        with input_container:
            with st.form(key="my_form", clear_on_submit=True):
                user_input = st.text_input(
                    "Ask a question",
                    placeholder="Ask Brikshyan iDOC",
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
                            search_kwargs={"k": 5}
                        )

                        docs = retriever.invoke(user_input)

                        context = "\n\n".join(
                            [
                                f"Source: {doc.metadata.get('source', 'Unknown')}, "
                                f"Page: {doc.metadata.get('page', 'Unknown')}\n"
                                f"{doc.page_content}"
                                for doc in docs
                            ]
                        )

                        prompt = f"""
                        You are a helpful GovTech document assistant.

                        Answer the user's question using ONLY the context below.

                        Rules:
                        - If the answer is not in the context, say:
                        "I don't know based on the uploaded documents."
                        - Mention source file names when useful.
                        - Be clear and practical.
                        - Do not invent information.

                        Context:
                        {context}

                        Question:
                        {user_input}

                        Answer:
                        """

                        response = llm.invoke(prompt)
                        answer = response.content

                loading_placeholder.empty()
                chat_bot_message_placeholder.markdown(answer)

            st.session_state["generated"].append(answer)
            st.session_state["history"].append((user_input, answer))