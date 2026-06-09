import os
from langchain.document_loaders import PyMuPDFLoader
from loadllm import Loadllm
from streamlit_chat import message
import tempfile
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import streamlit as st


DB_FAISS_PATH = 'vectorstore/db_faiss'

class FileIngestor:
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file

    def handlefileandingest(self):
        # Create temporary file to process uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(self.uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load embeddings using Sentence Transformers
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

        # Check if FAISS index already exists
        if os.path.exists(DB_FAISS_PATH + ".faiss") and os.path.exists(DB_FAISS_PATH + ".pkl"):
            # Load existing FAISS index
            db = FAISS.load_local(DB_FAISS_PATH, embeddings)
            st.info("Loaded existing FAISS index from disk.")
        else:
            # Load the document and create a new FAISS index
            loader = PyMuPDFLoader(file_path=tmp_file_path)
            data = loader.load()
            db = FAISS.from_documents(data, embeddings)
            db.save_local(DB_FAISS_PATH)
            #st.info("Created new FAISS index and saved to disk.")

        # Load the language model
        llm = Loadllm.load_llm()

        # Create a conversational chain
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

        # Function for conversational chat
        def conversational_chat(query):
            result = chain({"question": query, "chat_history": st.session_state['history'][-2:]})
            st.session_state['history'].append((query, result["answer"]))
            # Limit history to 2 by slicing the last two elements
            st.session_state['history'] = st.session_state['history'][-2:]
            return result["answer"]

        # Initialize chat history
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        # Initialize messages
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello! Ask me about " + self.uploaded_file.name + " ðŸ¤—"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey! ðŸ‘‹"]

        # Create containers for chat history and user input
        response_container = st.container()
        container = st.container()

        # User input form
        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Talk to PDF data ðŸ§®", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        # Display chat history
        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="initials", seed="Santosh Subedi")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="initials", seed="AI")
