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
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file

    def handlefileandingest(self):

        # Create temporary file to process uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(self.uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            print("tmp_file_path:", tmp_file_path)

        # Load embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load or create FAISS index
        if os.path.exists(DB_FAISS_PATH + ".faiss") and os.path.exists(DB_FAISS_PATH + ".pkl"):
            db = FAISS.load_local(
                DB_FAISS_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            st.info("Loaded existing FAISS index from disk.")
        else:
            loader = PyPDFLoader(file_path=tmp_file_path)
            data = loader.load()

            # Check empty PDF text
            if not data or not any(doc.page_content.strip() for doc in data):
                st.error("No readable text found in this PDF. It may be scanned or image-based.")
                return

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50
            )

            chunks = splitter.split_documents(data)

            # Remove empty chunks
            chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

            if not chunks:
                st.error("No valid text chunks found in the PDF.")
                return

            db = FAISS.from_documents(chunks, embeddings)
            db.save_local(DB_FAISS_PATH)

        # Initialize session state
        if "history" not in st.session_state:
            st.session_state["history"] = []

        if "generated" not in st.session_state:
            st.session_state["generated"] = [
                "Hello! Ask me about " + self.uploaded_file.name
            ]

        if "past" not in st.session_state:
            st.session_state["past"] = ["Hey! iDOC AI"]

        # Create containers
        chat_history_container = st.container()
        live_stream_container = st.container()
        input_container = st.container()

        # Display past chat
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

        # Live response placeholders
        with live_stream_container:
            loading_placeholder = st.empty()
            chat_bot_message_placeholder = st.empty()

        # Input form
        with input_container:
            with st.form(key="my_form", clear_on_submit=True):
                user_input = st.text_input(
                    "",
                    placeholder="Ask Brikshyan iDOC",
                    key="input"
                )
                submit_button = st.form_submit_button(label="Send")

        # Handle submit
        if submit_button and user_input:
            st.session_state["past"].append(user_input)

            # Show current user query immediately
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

                        # qa = RetrievalQA.from_chain_type(
                        #     llm=llm,
                        #     retriever=db.as_retriever(
                        #         search_type="similarity",
                        #         search_kwargs={"k": 5}
                        #     ),
                        #     return_source_documents=True
                        # )

                        # result = qa.invoke({"query": user_input})
                        # answer = result["result"]

                        retriever = db.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 5}
                        )

                        docs = retriever.invoke(user_input)

                        context = "\n\n".join([doc.page_content for doc in docs])

                        prompt = f"""
                        You are a helpful document assistant.

                        Answer the user's question using ONLY the context below.
                        If the answer is not in the context, say:
                        "I don't know based on the uploaded document."

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

            # Save chat history
            st.session_state["generated"].append(answer)
            st.session_state["history"].append((user_input, answer))