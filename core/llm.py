"""One place that talks to Anthropic or AWS Bedrock, so it can be turned off.

`complete_json` returns a parsed dict, or None on any failure at all -- no key,
no SDK, rate limit, timeout, malformed JSON. Every caller treats None as
"degrade to the deterministic path". That is what makes the demo survive a
dead conference network, and it is why nothing else in the codebase imports
`anthropic` directly.
"""

from __future__ import annotations

import json
import os
import re

DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5")
DEFAULT_MAX_TOKENS = 1200
DEFAULT_TIMEOUT = float(os.environ.get("AGENT_TIMEOUT", "30"))

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def available() -> bool:
    """True when an API key and the SDK are both present."""
    if _provider() == "bedrock":
        if not os.environ.get("BEDROCK_MODEL_ID"):
            return False
        try:
            import boto3  # noqa: F401
        except ImportError:
            return False
        return True
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _client():
    import anthropic

    return anthropic.Anthropic(timeout=DEFAULT_TIMEOUT)


def _provider() -> str:
    explicit = os.environ.get("AGENT_LLM_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    return "bedrock" if os.environ.get("BEDROCK_MODEL_ID") else "anthropic"


def _bedrock_text(system: str, user: str, model: str | None, max_tokens: int) -> str:
    import boto3

    session = boto3.Session(
        profile_name=os.environ.get("AWS_PROFILE") or None,
        region_name=(
            os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        ),
    )
    client = session.client("bedrock-runtime")
    response = client.converse(
        modelId=model or os.environ["BEDROCK_MODEL_ID"],
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": user}]}],
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": float(os.environ.get("AGENT_TEMPERATURE", "0.2")),
        },
    )
    return "".join(
        block.get("text", "")
        for block in response["output"]["message"]["content"]
    )


def complete_json(system: str, user: str, model: str | None = None,
                  max_tokens: int = DEFAULT_MAX_TOKENS) -> dict | None:
    """Ask for JSON and return it parsed, or None if anything goes wrong."""
    if os.environ.get("AGENT_LLM", "").strip().lower() == "off" or not available():
        return None
    try:
        if _provider() == "bedrock":
            text = _bedrock_text(system, user, model, max_tokens)
        else:
            response = _client().messages.create(
                model=model or DEFAULT_MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(
                block.text
                for block in response.content
                if getattr(block, "type", "") == "text"
            )
    except Exception:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(text)  # tolerate prose or fences around the JSON
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
