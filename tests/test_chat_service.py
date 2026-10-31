"""Tests for the customer-service prompt and streaming chat service (Task 4)."""

import pytest
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGenerationChunk

from app.prompts.customer_service import CUSTOMER_SERVICE_SYSTEM_PROMPT
from app.services.chat import assemble_messages, stream_chat, store
from app.services.memory import SessionStore


def test_customer_service_prompt_has_role_and_constraints():
    # 标注样例断言（非 TDD）：prompt 文本必须包含角色词与行为约束词
    assert "客服" in CUSTOMER_SERVICE_SYSTEM_PROMPT
    assert "不编造" in CUSTOMER_SERVICE_SYSTEM_PROMPT
    assert "人工" in CUSTOMER_SERVICE_SYSTEM_PROMPT


def test_assemble_messages_injects_system_and_trims(monkeypatch):
    # 预置超限历史：10 轮（20 条）> context_max_messages=5
    monkeypatch.setattr("app.config.settings.context_max_messages", 5)
    monkeypatch.setattr("app.config.settings.context_max_tokens", 100000)

    fake_store = SessionStore()
    for i in range(10):
        fake_store.append("s", HumanMessage(f"user-{i}"))
        fake_store.append("s", AIMessage(f"ai-{i}"))
    monkeypatch.setattr("app.services.chat.store", fake_store)

    messages = assemble_messages("s", "最新的问题")

    # 首条是注入的 SystemMessage，末条是新的 HumanMessage，中间是裁剪后历史
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[-1], HumanMessage)
    assert messages[-1].content == "最新的问题"
    middle = messages[1:-1]
    assert [m.content for m in middle] == ["ai-7", "user-8", "ai-8", "user-9", "ai-9"]


class _EmptyReplyModel(SimpleChatModel):
    """Fake model that yields a single empty-content chunk (empty reply)."""

    @property
    def _llm_type(self) -> str:
        return "empty-reply-model"

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        return ""

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        yield ChatGenerationChunk(message=AIMessageChunk(content="", chunk_position="last"))


@pytest.mark.asyncio
async def test_stream_chat_yields_tokens_one_by_one(monkeypatch):
    monkeypatch.setattr(
        "app.services.chat.get_chat_model",
        lambda: FakeListChatModel(responses=["你好啊"]),
    )
    monkeypatch.setattr("app.services.chat.store", SessionStore())

    tokens = [tok async for tok in stream_chat("s", "hi")]
    assert tokens == ["你", "好", "啊"]


@pytest.mark.asyncio
async def test_stream_chat_yields_once_on_empty_reply(monkeypatch):
    monkeypatch.setattr("app.services.chat.get_chat_model", lambda: _EmptyReplyModel())
    monkeypatch.setattr("app.services.chat.store", SessionStore())

    tokens = [tok async for tok in stream_chat("s", "hi")]
    assert tokens == [""]
