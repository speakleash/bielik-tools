"""Shared vLLM client config for examples (v3.1 chat template + parsers).

Reads (in order): process env, then optional sibling/parent .env files.
Defaults target a local server serving model id ``step72``.
"""
from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI

_ENV_KEYS = ("VLLM_BASE_URL", "VLLM_API_KEY", "VLLM_MODEL")
_DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
_DEFAULT_MODEL = "step72"


def _load_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        out[key] = value
    return out


def resolve_vllm_config() -> dict[str, str]:
    merged: dict[str, str] = {}
    examples_dir = Path(__file__).resolve().parent
    for candidate in (
        examples_dir / ".env",
        examples_dir.parent / ".env",
        examples_dir.parent.parent / "bielik-chat-templates" / ".env",
    ):
        merged.update(_load_dotenv(candidate))
    for key in _ENV_KEYS:
        if os.environ.get(key):
            merged[key] = os.environ[key]

    base_url = (merged.get("VLLM_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
    api_key = merged.get("VLLM_API_KEY") or "EMPTY"
    model = merged.get("VLLM_MODEL") or _DEFAULT_MODEL
    # Local smoke servers often expose a short served name (e.g. step72).
    # Prefer explicit env; otherwise keep default.
    return {"base_url": base_url, "api_key": api_key, "model": model}


def make_client() -> tuple[OpenAI, str]:
    cfg = resolve_vllm_config()
    client = OpenAI(api_key=cfg["api_key"] or "EMPTY", base_url=cfg["base_url"])
    return client, cfg["model"]
