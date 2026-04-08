import os

API_BASE_URL: str = os.getenv("API_BASE_URL", "")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:7860")
APP_PORT: int = int(os.getenv("PORT", "7860"))
