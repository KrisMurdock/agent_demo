import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()


def get_model():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o"),
        api_key=os.getenv("OPENAI_API_KEY", "sk-placeholder"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        temperature=float(os.getenv("TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
    )
