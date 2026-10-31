"""Tests for the FastAPI routes and app entrypoint (Task 6)."""

import app.routes.chat as chat_routes
from langchain_core.messages import AIMessage, HumanMessage


def test_chat_sse_stream(client_with_fake_llm):
    with client_with_fake_llm.stream(
        "POST", "/chat", json={"session_id": "s1", "message": "你好"}
    ) as r:
        body = "".join(r.iter_text())
    assert r.status_code == 200
    assert "data: " in body
    assert "[DONE]" in body
    assert body.strip().endswith("data: [DONE]")


def test_chat_persists_user_and_assistant(client_with_fake_llm):
    """R1: after the stream completes, BOTH the user turn and the assistant
    reply are persisted, so round 2 sees round 1's context (no duplicate user)."""
    with client_with_fake_llm.stream(
        "POST", "/chat", json={"session_id": "s1", "message": "你好"}
    ) as r:
        body = "".join(r.iter_text())
    assert r.status_code == 200
    assert "[DONE]" in body

    history = chat_routes.store.get_history("s1")
    assert [type(m) for m in history] == [HumanMessage, AIMessage]
    assert history[0].content == "你好"
    assert history[1].content == "你好，欢迎光临"


def test_extract_json(client_with_fake_llm):
    r = client_with_fake_llm.post("/extract", json={"text": "订单号 1 要退货"})
    assert r.status_code == 200
    assert set(r.json().keys()) == {"order_id", "request_type", "expected_solution"}
