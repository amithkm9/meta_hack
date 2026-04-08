import os


API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
APP_PORT: int = int(os.getenv("PORT", "7860"))
