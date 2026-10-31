def test_settings_defaults():
    from app.config import Settings

    s = Settings(_env_file=None)
    assert s.llm_provider == "deepseek"
    assert s.llm_model == "deepseek-chat"
    assert s.context_max_tokens == 4000


def test_settings_from_env(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("LLM_MODEL", "deepseek-reasoner")
    monkeypatch.setenv("CONTEXT_MAX_TOKENS", "100")
    s = Settings(_env_file=None)
    assert s.llm_model == "deepseek-reasoner"
    assert s.context_max_tokens == 100
