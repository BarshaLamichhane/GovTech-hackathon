import os
import tempfile

import streamlit as st
from streamlit_chat import message

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from loadllm import Loadllm

DB_FAISS_PATH = 'vectorstore/db_faiss'

class FileIngestorV2:
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file

    def handlefileandingest(self):
        
        # Create temporary file to process uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(self.uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            print("tmp_file_path",tmp_file_path)
        

        # Load embeddings using Sentence Transformers
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

        # Check if FAISS index already exists
        if os.path.exists(DB_FAISS_PATH + ".faiss") and os.path.exists(DB_FAISS_PATH + ".pkl"):
            # Load existing FAISS index
            try:
                db = FAISS.load_local(
                    DB_FAISS_PATH,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
            except TypeError:
                db = FAISS.load_local(DB_FAISS_PATH, embeddings)
            st.info("Loaded existing FAISS index from disk.")
        else:
            # Load the document and create a new FAISS index
            loader = PyPDFLoader(file_path=tmp_file_path)
            data = loader.load()
            # print("Loaded data:", data)
            # print("Number of docs loaded:", len(data))
            
            #db = FAISS.from_documents(data, embeddings)

            ##Added later started###########
            splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
            chunks = splitter.split_documents(data)
            db = FAISS.from_documents(chunks, embeddings)
            ##Added later ended###########

            db.save_local(DB_FAISS_PATH)
            #st.info("Created new FAISS index and saved to disk.")

       
    
        
       

        

        # Load the language model
        # Create a conversational chain
        #chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

            

        # Initialize chat history
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        else:
            print("history is not empty", st.session_state['history'])
        # Initialize messages
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello! Ask me about " + self.uploaded_file.name ]
        else:
            print("genereted is not empty", st.session_state['generated'])

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey! iDOC AI"]
        else:
            print("past is not empty", st.session_state['past'])
        

        # Create containers for chat history, input query and live stream
        chat_history_container = st.container()
        live_stream_container = st.container()
        input_container = st.container()

        ############### Past chat ###############
        with chat_history_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=f"{i}_user", avatar_style="initials", seed="User")
                message(st.session_state["generated"][i], key=f"{i}", avatar_style="initials", seed="Brikshyan Solutions")

       ############### Live stream placeholder ###############
        
        with live_stream_container:
            loading_placeholder = st.empty() ### place holder for loading giphy
            chat_bot_message_placeholder = st.empty() ### place holder for live streaming chatbot message holder

        ############### Input query Form ###############
        with input_container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("", placeholder="Ask Brikshyan iDOC", key='input')
                submit_button = st.form_submit_button(label='Send')

        ############### Show full chat history ###############
        if submit_button and user_input:
            st.session_state['past'].append(user_input)
            
            ##########This will show the input query immediately after asking; if avoided this line input query will be shown as chat history after entering another input query not as current query immediately. ##########
            with chat_history_container:
                message(user_input, is_user=True, key=f"{len(st.session_state['past'])}_user", avatar_style="initials", seed="User")

            #loading_placeholder.markdown("Thinking")
            # llm = Loadllm.load_llm(container=chat_bot_message_placeholder)

            # qa = RetrievalQA.from_chain_type(
            #     llm=llm,
            #     retriever=db.as_retriever(search_type="similarity", k=5),
            #     return_source_documents=True
            # )

            # result = qa({"query": user_input, "chat_history": st.session_state['history'][-2:]})
            # answer = result["result"]


            with live_stream_container:

                # Show spinner while processing
                with loading_placeholder.container():
                    with st.spinner("🧠 Thinking ... ..."):
                       
                        # Load LLM with live streaming output
                        llm = Loadllm.load_llm(container=chat_bot_message_placeholder)

                        qa = RetrievalQA.from_chain_type(
                            llm=llm,
                            retriever=db.as_retriever(search_type="similarity", k=5),
                            return_source_documents=True
                        )

                        result = qa({"query": user_input, "chat_history": st.session_state['history'][-2:]})
                        answer = result["result"]

                

                loading_placeholder.empty() ## clear spinner after getting full answer
                chat_bot_message_placeholder.markdown(answer) ## This will remove the blinking cursor after full answer is generated.
            
                

            ############This will save current chat history so that it would be displayed after entering another query.##################
            st.session_state['generated'].append(answer)
            st.session_state['history'].append((user_input, answer))
