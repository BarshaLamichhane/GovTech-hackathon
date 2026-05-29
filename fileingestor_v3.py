import os
from langchain.document_loaders import PyPDFLoader
from loadllm import Loadllm
from streamlit_chat import message
import tempfile
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import streamlit as st 

##Added later started###########
#pdf loader
from langchain.text_splitter import RecursiveCharacterTextSplitter

##Embedding and faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

###Retrivaé and LLM Prompting
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
### added later ended ############################################
DB_FAISS_PATH = 'vectorstore/db_faiss'

class FileIngestorV2:
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file

    def handlefileandingest(self):
        print("ccccccccccccccccc")
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
            loader = PyPDFLoader(file_path=tmp_file_path)
            data = loader.load()
            
            #db = FAISS.from_documents(data, embeddings)

            ##Added later started###########
            splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
            chunks = splitter.split_documents(data)
            db = FAISS.from_documents(chunks, embeddings)
            ##Added later ended###########

            db.save_local(DB_FAISS_PATH)
            #st.info("Created new FAISS index and saved to disk.")

        
        
        llm = Loadllm.load_llm()

        ##Added later started###########
        # Load the language model
       
        qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_type="similarity", k=5),
        return_source_documents=True
        )
        ##Added later ended###########

        # Load the language model
        # Create a conversational chain
        #chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

        # Function for conversational chat
        def conversational_chat(query):
            print("Query:", query)
            print("History:", st.session_state['history'][-2:])
            #result = chain({"question": query, "chat_history": st.session_state['history'][-2:]})
            result = qa({"query": query, "chat_history": st.session_state['history'][-2:]})
            print("BEFORE RESULT IS", result)
            st.session_state['history'].append((query, result["result"]))
            #st.session_state['history'].append((query, result["answer"]))
            # Limit history to 2 by slicing the last two elements
            st.session_state['history'] = st.session_state['history'][-2:]
            print("After result is",result)
            #return result["answer"]
            return result["result"]


        #######################################################################

        
        # ##################################################################    

        # Initialize chat history
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        # Initialize messages
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello! Ask me about " + self.uploaded_file.name + " abccd"]

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
                print("I am past")
                st.session_state['generated'].append(output)
                print("I am generated")
                print(output)
            else:
                print("something error")

        # Display chat history
        if st.session_state['generated']:
            print("hello I am generated")
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="initials", seed="Santosh Subedi")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="initials", seed="AI")
        else:
            print("oops session is not generated")

        