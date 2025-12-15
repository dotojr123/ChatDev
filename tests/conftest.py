import pytest
from chatdev.settings import settings

@pytest.fixture
def mock_settings(monkeypatch):
    monkeypatch.setattr(settings, "QDRANT_URL", ":memory:")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-mock-key")
    return settings
