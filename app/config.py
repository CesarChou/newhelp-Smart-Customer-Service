"""Application configuration via pydantic-settings.

A single :class:`Settings` instance is exported as ``settings`` and consumed
by every later task. Values come from environment variables (or a ``.env``
file), falling back to the defaults defined here.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the e-commerce customer-service system."""

    llm_provider: str = "deepseek"
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    llm_api_key: str = ""
    context_max_tokens: int = 4000
    context_max_messages: int = 20
    chars_per_token: float = 1.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
