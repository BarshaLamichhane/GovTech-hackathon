import os
from dotenv import load_dotenv

load_dotenv()

OFFLINE_MODEL_PATH = os.getenv("OFFLINE_MODEL_PATH", "")
MISTRAL_API_MODEL_NAME = os.getenv("MISTRAL_API_MODEL_NAME")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_MODEL_NAME = os.getenv("GEMINI_API_MODEL_NAME")


class LoadLLM:
    @staticmethod
    def load_llm():
        if OFFLINE_MODEL_PATH and os.path.exists(OFFLINE_MODEL_PATH):
            return LoadLLM.load_local_model()

        if GEMINI_API_KEY and GEMINI_API_MODEL_NAME:
            return LoadLLM.load_online_gemini()

        if MISTRAL_API_KEY and MISTRAL_API_MODEL_NAME:
            return LoadLLM.load_online_mistral()

        raise RuntimeError(
            "No model configuration found. Set OFFLINE_MODEL_PATH or MISTRAL_API_KEY / GEMINI_API_KEY."
        )

    @staticmethod
    def load_local_model():
        try:
            from langchain.llms import LlamaCpp
        except ImportError as exc:
            raise RuntimeError("llama-cpp-python is required for offline model support") from exc

        return LlamaCpp(model_path=OFFLINE_MODEL_PATH, temperature=0)

    @staticmethod
    def load_online_mistral():
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(
            model=MISTRAL_API_MODEL_NAME,
            temperature=0,
            max_retries=2,
            api_key=MISTRAL_API_KEY,
        )

    @staticmethod
    def load_online_gemini():
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=GEMINI_API_MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=0,
            max_retries=2,
        )
