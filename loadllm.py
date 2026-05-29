# from langchain.llms import LlamaCpp
# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.callbacks.base import BaseCallbackHandler
# import os
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_mistralai import ChatMistralAI


# model_name="llama-2-7b-chat.Q4_K_M.gguf"
# model_path = os.path.join("models", model_name)
# #'models\llama-2-7b-chat.ggmlv3.q2_K.bin'

# load_dotenv()

# # Read variables
# MISTRAL_API_MODEL_NAME = os.getenv("MISTRAL_API_MODEL_NAME")
# MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# class StreamlitCallbackHandler(BaseCallbackHandler):
#     # def __init__(self, container):
#     #     self.container = container
#     #     self.text = ""

    
#     # def on_llm_new_token(self, token: str, **kwargs) -> None:
#     #     self.text += token
#     #     self.container.markdown(self.text + "▌")  # live typing effect

   
#     def __init__(self, container_placeholder):
#         self.container_placeholder = container_placeholder
#         self.text = ""

#     def on_llm_new_token(self, token: str, **kwargs):
#         self.text += token
#         self.container_placeholder.markdown(self.text + "▌")  # ▌ for blinking effect


# # class Loadllm:
# #     @staticmethod
# #     #@st.cache_resource
# #     def load_llm(container=None):
# #         callback_manager = CallbackManager([
# #             StreamingStdOutCallbackHandler(),                      # outputs in terminal console
# #             StreamlitCallbackHandler(container) if container else None  # outputs in Streamlit
# #         ])
# #         # Prepare the LLM

# #         llm = LlamaCpp(
# #             model_path=model_path,
# #             n_gpu_layers=40,
# #             n_batch=512,
# #             n_ctx=2048,
# #             f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
# #             callback_manager=callback_manager, #writing output to console.
# #             verbose=True,
# #             streaming=True #important for streaming directly in streamlit app.

# #         )

# #         return llm

# class Loadllm:
#     @staticmethod
#     def load_llm_llama(container=None):
#         handlers = [StreamingStdOutCallbackHandler()]
#         if container:
#             handlers.append(StreamlitCallbackHandler(container))

#         callback_manager = CallbackManager(handlers)

#         llm = LlamaCpp(
#             model_path=model_path,
#             n_gpu_layers=40,
#             n_batch=512,
#             n_ctx=2048,
#             f16_kv=True,
#             callback_manager=callback_manager,
#             verbose=True,
#             streaming=True
#         )

#         return llm
    
#     @staticmethod
#     def load_llm():
#         llm = ChatMistralAI(
#             model=MISTRAL_API_MODEL_NAME,
#             temperature=0,
#             max_retries=2,
#             api_key=MISTRAL_API_KEY
#         )
#         return llm

import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
MISTRAL_API_MODEL_NAME = os.getenv("MISTRAL_API_MODEL_NAME")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_MODEL_NAME = os.getenv("GEMINI_API_MODEL_NAME")

class Loadllm:
    @staticmethod
    def load_llm():
        llm = ChatMistralAI(
            model=MISTRAL_API_MODEL_NAME,
            temperature=0,
            max_retries=2,
            api_key=MISTRAL_API_KEY
        )
        return llm
    
    @staticmethod
    def load_gemini_llm():
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_API_MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=0,
            max_retries=2
        )
        return llm