"""
config.py
---------
Central place that reads .env and exposes typed settings to the rest of
the backend. Nothing else in backend/ should call os.environ directly —
import `settings` from here instead.

Why pydantic-settings: it validates types (e.g. catches a typo'd env var
early), gives IDE autocomplete, and is what FastAPI projects use by
convention — worth a line in the report.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_provider: str = "gemini"

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash"

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
