import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()


def get_model():
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model=os.getenv("MODEL_NAME", "gpt-4o"),
        model_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY", "sk-placeholder"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        temperature=float(os.getenv("TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
    )
