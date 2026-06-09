import os

from settings import settings


class LoadLLM:
    @staticmethod
    def load_llm():
        provider = settings.llm_provider
        offline_model_path = os.getenv("OFFLINE_MODEL_PATH", "")

        if provider == "azure":
            return LoadLLM.load_azure_openai()
        if provider == "local":
            return LoadLLM.load_local_model()
        if provider == "gemini":
            return LoadLLM.load_online_gemini()
        if provider == "mistral":
            return LoadLLM.load_online_mistral()
        if provider != "auto":
            raise RuntimeError("LLM_PROVIDER must be auto, local, azure, mistral, or gemini.")

        if settings.azure_openai_endpoint and settings.azure_openai_chat_deployment:
            return LoadLLM.load_azure_openai()
        if offline_model_path and os.path.exists(offline_model_path):
            return LoadLLM.load_local_model()
        if os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_API_MODEL_NAME"):
            return LoadLLM.load_online_gemini()
        if os.getenv("MISTRAL_API_KEY") and os.getenv("MISTRAL_API_MODEL_NAME"):
            return LoadLLM.load_online_mistral()

        raise RuntimeError(
            "No model configuration found. Configure Azure OpenAI, OFFLINE_MODEL_PATH, Mistral, or Gemini."
        )

    @staticmethod
    def load_local_model():
        model_path = os.getenv("OFFLINE_MODEL_PATH", "")
        if not model_path or not os.path.exists(model_path):
            raise RuntimeError("OFFLINE_MODEL_PATH must point to an existing GGUF model.")
        try:
            from langchain_community.llms import LlamaCpp
        except ImportError as exc:
            raise RuntimeError("llama-cpp-python is required for offline model support") from exc

        return LlamaCpp(model_path=model_path, temperature=0)

    @staticmethod
    def load_azure_openai():
        if not settings.azure_openai_endpoint or not settings.azure_openai_chat_deployment:
            raise RuntimeError(
                "Azure OpenAI requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_CHAT_DEPLOYMENT."
            )

        from langchain_openai import AzureChatOpenAI

        kwargs = {
            "azure_endpoint": settings.azure_openai_endpoint,
            "azure_deployment": settings.azure_openai_chat_deployment,
            "api_version": settings.azure_openai_api_version,
            "temperature": 0,
            "max_retries": 2,
        }
        if settings.azure_openai_api_key:
            kwargs["api_key"] = settings.azure_openai_api_key
        else:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default",
            )

        return AzureChatOpenAI(**kwargs)

    @staticmethod
    def load_online_mistral():
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(
            model=os.getenv("MISTRAL_API_MODEL_NAME"),
            temperature=0,
            max_retries=2,
            api_key=os.getenv("MISTRAL_API_KEY"),
        )

    @staticmethod
    def load_online_gemini():
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_API_MODEL_NAME"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0,
            max_retries=2,
        )
