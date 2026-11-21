"""Runtime configuration loaded from environment variables / backend/.env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional
    load_dotenv = None

ROOT = Path(__file__).resolve().parents[1]
if load_dotenv:
    load_dotenv(ROOT / ".env")

# OpenAI-compatible chat endpoints. Groq is free with an API key; Ollama runs locally.
PROVIDERS: dict[str, dict[str, object]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"],
        "key_env": "GROQ_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "models": ["llama3.1:8b"],
        "key_env": None,
    },
}


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    base_url: str
    models: list[str]
    api_key: str | None
    timeout: float

    @property
    def enabled(self) -> bool:
        if self.provider not in PROVIDERS:
            return False
        if PROVIDERS[self.provider]["key_env"] and not self.api_key:
            return False
        return True


def llm_settings() -> LLMSettings:
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    preset = PROVIDERS.get(provider, {"base_url": "", "models": [], "key_env": None})
    key_env = preset["key_env"]
    custom_model = os.getenv("LLM_MODEL", "").strip()
    models = list(preset["models"])  # type: ignore[arg-type]
    if custom_model:
        models = [custom_model] + [m for m in models if m != custom_model]
    return LLMSettings(
        provider=provider,
        base_url=os.getenv("LLM_BASE_URL", str(preset["base_url"])).rstrip("/"),
        models=models,
        api_key=os.getenv(str(key_env)) if key_env else None,
        timeout=float(os.getenv("LLM_TIMEOUT", "60")),
    )


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDINGS_ENABLED = os.getenv("EMBEDDINGS_ENABLED", "true").strip().lower() not in {"0", "false", "no"}
