"""Minimal client for OpenAI-compatible chat APIs (Groq, Ollama)."""

from __future__ import annotations

import json
import logging
import re

import httpx

from app.config import llm_settings

log = logging.getLogger("researchai.llm")


class LLMError(RuntimeError):
    pass


def llm_status() -> dict[str, object]:
    settings = llm_settings()
    return {
        "provider": settings.provider,
        "enabled": settings.enabled,
        "model": settings.models[0] if settings.models else None,
    }


def chat(messages: list[dict[str, str]], temperature: float = 0.2, max_tokens: int = 1200) -> str:
    """Send a chat request, trying each configured model until one works."""
    settings = llm_settings()
    if not settings.enabled:
        raise LLMError("LLM is not configured.")

    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    last_error = "unknown error"
    for model in settings.models:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            response = httpx.post(
                f"{settings.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=settings.timeout,
            )
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            log.warning("LLM request failed for %s: %s", model, last_error)
            continue

        if response.status_code == 200:
            data = response.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            if content:
                return content
            last_error = f"{model} returned an empty response"
            continue

        last_error = f"{model} -> HTTP {response.status_code}: {response.text[:300]}"
        log.warning("LLM error: %s", last_error)
        if response.status_code in (401, 403):
            break  # bad key: other models will fail too
    raise LLMError(last_error)


def chat_json(messages: list[dict[str, str]], max_tokens: int = 1500) -> object:
    """Ask for JSON output and parse it, tolerating ```json fences."""
    text = chat(messages, temperature=0.1, max_tokens=max_tokens)
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        text = match.group(1)
    start = min([i for i in (text.find("{"), text.find("[")) if i != -1], default=0)
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError as exc:
        raise LLMError(f"Could not parse JSON from LLM: {exc}") from exc
