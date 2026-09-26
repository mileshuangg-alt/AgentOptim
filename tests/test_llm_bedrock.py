"""Bedrock uses the same complete_json boundary as the existing agents."""

from __future__ import annotations

from core import llm


def test_bedrock_provider_parses_json(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("BEDROCK_MODEL_ID", "test-model")
    monkeypatch.delenv("AGENT_LLM", raising=False)
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(
        llm,
        "_bedrock_text",
        lambda system, user, model, max_tokens: (
            '```json\n{"verdict":"support","preferred":"CCO"}\n```'
        ),
    )
    assert llm.complete_json("system", "user") == {
        "verdict": "support",
        "preferred": "CCO",
    }
