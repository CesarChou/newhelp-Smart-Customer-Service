"""Shared pytest fixtures for the ecom-cs test suite.

Task 1 (scaffolding) needs no fixtures yet; later tasks register their fakes
here. Task 6 registers ``client_with_fake_llm``: a ``TestClient`` wired to the
real ``app.main.app`` with ``stream_chat``/``extract`` monkeypatched to fakes
(and a fresh session store), so route tests never hit a real network.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.extraction import AfterSaleExtraction, RequestType
from app.services.memory import SessionStore


@pytest.fixture
def client_with_fake_llm(monkeypatch):
    """TestClient with fake chat/extraction services and a clean session store."""

    async def fake_stream_chat(session_id, user_input):
        yield "你好"
        yield "，欢迎光临"

    async def fake_extract(text):
        return AfterSaleExtraction(
            order_id="1", request_type=RequestType.退货, expected_solution="退款"
        )

    monkeypatch.setattr("app.routes.chat.stream_chat", fake_stream_chat)
    monkeypatch.setattr("app.routes.chat.store", SessionStore())
    monkeypatch.setattr("app.routes.extraction.extract", fake_extract)

    with TestClient(app) as client:
        yield client
