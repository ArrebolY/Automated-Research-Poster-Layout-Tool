from __future__ import annotations

import json
import http.client
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .config import LLMConfig


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str:
        ...


@dataclass(slots=True)
class NoOpTextGenerator:
    def generate(self, prompt: str) -> str:
        raise RuntimeError("No LLM is configured. Use provider='rule' for the offline baseline.")


@dataclass(slots=True)
class OpenAICompatibleTextGenerator:
    config: LLMConfig

    def generate(self, prompt: str) -> str:
        api_key = self._api_key()
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You convert research paper sections into concise academic poster bullet points.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
        }
        if self.config.provider.lower() == "deepseek":
            # Structured poster tasks need a short final answer, not a long reasoning trace.
            payload["thinking"] = {"type": "disabled"}
        data = self._post_json(self._endpoint(), payload, {"Authorization": f"Bearer {api_key}"})
        content = data["choices"][0]["message"].get("content") or ""
        content = content.strip()
        if not content:
            raise RuntimeError("The LLM returned an empty final response.")
        return content

    def _api_key(self) -> str:
        env_name = self.config.api_key_env or _default_api_key_env(self.config.provider)
        api_key = os.getenv(env_name, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing API key. Set the {env_name} environment variable.")
        return api_key

    def _endpoint(self) -> str:
        base_url = self.config.base_url or _default_base_url(self.config.provider)
        return f"{base_url.rstrip('/')}/chat/completions"

    def _post_json(self, url: str, payload: dict, headers: dict[str, str]) -> dict:
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(4):
            request = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json", **headers},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                if exc.code == 402:
                    raise RuntimeError(
                        "LLM request failed because the provider account has insufficient balance. "
                        "Please top up the account or switch to another provider/model."
                    ) from exc
                if exc.code not in {408, 409, 425, 429, 500, 502, 503, 504}:
                    raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {details}") from exc
                last_error = RuntimeError(f"LLM request failed with HTTP {exc.code}: {details}")
            except (http.client.RemoteDisconnected, TimeoutError, urllib.error.URLError, OSError) as exc:
                last_error = exc
            if attempt < 3:
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM request failed after retries: {type(last_error).__name__}: {last_error}") from last_error


@dataclass(slots=True)
class GeminiTextGenerator:
    config: LLMConfig

    def generate(self, prompt: str) -> str:
        env_name = self.config.api_key_env or "GEMINI_API_KEY"
        api_key = os.getenv(env_name, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing API key. Set the {env_name} environment variable.")

        model = self.config.model or "gemini-3.1-flash-lite"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_output_tokens,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini request failed with HTTP {exc.code}: {details}") from exc

        parts = data["candidates"][0]["content"]["parts"]
        return "\n".join(part.get("text", "") for part in parts).strip()


def build_text_generator(config: LLMConfig) -> TextGenerator | None:
    provider = config.provider.lower()
    if provider == "rule":
        return None
    if provider in {"openai", "deepseek", "openrouter"}:
        return OpenAICompatibleTextGenerator(config)
    if provider == "gemini":
        return GeminiTextGenerator(config)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")


def _default_api_key_env(provider: str) -> str:
    defaults = {
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    return defaults.get(provider.lower(), "LLM_API_KEY")


def _default_base_url(provider: str) -> str:
    defaults = {
        "openai": "https://api.openai.com/v1",
        "deepseek": "https://api.deepseek.com",
        "openrouter": "https://openrouter.ai/api/v1",
    }
    return defaults[provider.lower()]
