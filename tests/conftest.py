"""Shared fixtures: unit tests must never reach the OpenAI API."""
import pytest

from rag import providers


@pytest.fixture(autouse=True)
def block_openai(monkeypatch):
    """Replace the OpenAI client factory so that any accidental API call fails instead of spending money."""

    def blocked():
        """Raise instead of creating a client."""
        raise RuntimeError("unit tests must not call the OpenAI API")

    monkeypatch.setattr(providers, "_client", blocked)
