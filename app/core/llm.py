"""Model factory.

Builds the configured chat model and its structured-output wrapper.
Later tasks (chat service, extraction service) depend on the exact
signatures exported here:

* :func:`get_chat_model` -> ``BaseChatModel``
* :func:`get_structured_llm` -> ``Runnable``
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from app.config import settings

_OPENAI_COMPAT_PROVIDERS = ("openai", "deepseek", "ollama")

# langchain-openai 1.7.x constructs its OpenAI clients eagerly and refuses an
# empty/missing api key. When no key is configured we still need a constructed
# model (for DI wiring and factory tests); real calls will then fail with a
# clear auth error from the upstream.
_PLACEHOLDER_API_KEY = "sk-not-configured"


def get_chat_model() -> BaseChatModel:
    """Return a chat model matching ``settings.llm_provider``.

    ``openai``/``deepseek``/``ollama`` all route through ``ChatOpenAI``
    (OpenAI-compatible upstreams). ``anthropic`` uses ``ChatAnthropic``
    (imported lazily; not exercised in this environment). Any other value
    raises :class:`ValueError`.
    """
    provider = settings.llm_provider.lower()

    if provider in _OPENAI_COMPAT_PROVIDERS:
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": settings.llm_model,
            "api_key": settings.llm_api_key or _PLACEHOLDER_API_KEY,
            "base_url": settings.llm_base_url,
        }
        # OpenAI-compatible upstreams (e.g. DeepSeek) must hit the Chat
        # Completions endpoint, not the Responses API. Only pass the flag
        # when the installed langchain-openai exposes it.
        if "use_responses_api" in ChatOpenAI.model_fields:
            kwargs["use_responses_api"] = False
        return ChatOpenAI(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # 未实测 (untested)

        return ChatAnthropic(model=settings.llm_model, api_key=settings.llm_api_key)

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider!r}")


def get_structured_llm(schema: type[BaseModel]) -> Runnable:
    """Return ``get_chat_model()`` wrapped with structured output for ``schema``.

    Uses the default ``function_calling`` method.
    """
    return get_chat_model().with_structured_output(schema)
