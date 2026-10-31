"""Tests for the in-memory session store with token-budget trimming (Task 3)."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.services.memory import SessionStore


def test_get_history_empty_session():
    store = SessionStore()
    assert store.get_history("new-session") == []


def test_trim_keeps_system_and_latest_user(monkeypatch):
    monkeypatch.setattr("app.config.settings.context_max_tokens", 20)
    monkeypatch.setattr("app.config.settings.context_max_messages", 100)
    monkeypatch.setattr("app.config.settings.chars_per_token", 1.0)

    store = SessionStore()
    store.append("s", SystemMessage("sys"))          # 3 tokens, protected
    store.append("s", HumanMessage("olduser"))       # 7 tokens, oldest user -> dropped
    store.append("s", AIMessage("oldai"))            # 5 tokens, dropped
    store.append("s", HumanMessage("latestuser"))    # 10 tokens, latest user -> kept
    store.append("s", AIMessage("latestai"))         # 8 tokens, dropped

    history = store.get_history("s")

    # system + latest user survive; everything older is trimmed away
    assert [type(m).__name__ for m in history] == ["SystemMessage", "HumanMessage"]
    assert history[1].content == "latestuser"


def test_token_budget_zero_still_keeps_one_turn(monkeypatch):
    monkeypatch.setattr("app.config.settings.context_max_tokens", 0)
    monkeypatch.setattr("app.config.settings.context_max_messages", 100)
    monkeypatch.setattr("app.config.settings.chars_per_token", 1.0)

    store = SessionStore()
    store.append("s", SystemMessage("sys"))
    store.append("s", HumanMessage("what is my order?"))
    store.append("s", AIMessage("here it is"))

    history = store.get_history("s")

    # even with a zero budget the latest user turn must survive
    assert any(isinstance(m, HumanMessage) and m.content == "what is my order?" for m in history)
    assert any(isinstance(m, SystemMessage) for m in history)


def test_tail_trim_to_context_max_messages(monkeypatch):
    monkeypatch.setattr("app.config.settings.context_max_messages", 3)
    monkeypatch.setattr("app.config.settings.context_max_tokens", 100000)

    store = SessionStore()
    for i in range(5):
        store.append("s", HumanMessage(f"turn-{i}"))

    history = store.get_history("s")
    assert [m.content for m in history] == ["turn-2", "turn-3", "turn-4"]


def test_tail_trim_preserves_leading_system(monkeypatch):
    monkeypatch.setattr("app.config.settings.context_max_messages", 3)
    monkeypatch.setattr("app.config.settings.context_max_tokens", 100000)

    store = SessionStore()
    store.append("s", SystemMessage("sys"))
    for i in range(4):
        store.append("s", HumanMessage(f"turn-{i}"))

    history = store.get_history("s")
    # leading system message survives the tail slice and stays at the front
    assert history[0] == SystemMessage("sys")
    assert [m.content for m in history[1:]] == ["turn-1", "turn-2", "turn-3"]


def test_token_estimate_flattens_list_content(monkeypatch):
    monkeypatch.setattr("app.config.settings.context_max_tokens", 20)
    monkeypatch.setattr("app.config.settings.context_max_messages", 100)
    monkeypatch.setattr("app.config.settings.chars_per_token", 1.0)

    store = SessionStore()
    store.append("s", HumanMessage("q"))  # 1 token, latest user -> protected
    store.append("s", AIMessage(content=[{"type": "text", "text": "x" * 20}]))  # 20 tokens

    history = store.get_history("s")
    # total 21 > 20, so the trailing AI message is trimmed, leaving only the user
    assert len(history) == 1
    assert isinstance(history[0], HumanMessage)
