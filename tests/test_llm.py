"""Tests for the model factory in ``app.core.llm`` (Task 2)."""

import pytest
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from app.core.llm import get_chat_model, get_structured_llm


def test_get_chat_model_openai_compat(monkeypatch):
    # Global Constraint: the Chat Completions endpoint, not the Responses API.
    monkeypatch.setattr("app.config.settings.llm_provider", "deepseek")
    monkeypatch.setattr("app.config.settings.llm_model", "deepseek-chat")
    monkeypatch.setattr("app.config.settings.llm_base_url", "https://api.deepseek.com")
    model = get_chat_model()
    assert model.model_name == "deepseek-chat"
    base = getattr(model, "base_url", None) or getattr(model, "openai_api_base", None)
    assert base == "https://api.deepseek.com"
    assert model.use_responses_api is False


def test_get_chat_model_unknown_provider_raises(monkeypatch):
    monkeypatch.setattr("app.config.settings.llm_provider", "unknown")
    with pytest.raises(ValueError):
        get_chat_model()


class _Order(BaseModel):
    order_id: str
    total: float


def test_get_structured_llm_returns_runnable():
    runnable = get_structured_llm(_Order)
    assert isinstance(runnable, Runnable)
