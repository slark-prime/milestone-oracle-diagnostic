from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    timeout: float = 30000.0
    provider: Literal["openai_compatible", "openai", "prime", "tinker"] = "openai_compatible"

    PRIME_BASE_URL = "https://api.pinference.ai/api/v1"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LLMConfig":
        provider = str(payload.get("provider", "")).strip().lower()

        api_key = payload.get("api_key", "")
        api_key_env = payload.get("api_key_env")
        if not api_key and api_key_env:
            api_key = os.getenv(api_key_env, "")
        if not api_key and provider != "tinker":
            raise ValueError("Missing api_key or api_key_env for LLM config")

        base_url = payload.get("base_url", "")
        if not provider:
            provider = "openai" if "api.openai.com" in base_url else "openai_compatible"
        if provider == "prime":
            base_url = base_url or cls.PRIME_BASE_URL
        if not base_url and provider != "tinker":
            raise ValueError("Missing base_url for LLM config")
        if provider not in {"openai_compatible", "openai", "prime", "tinker"}:
            raise ValueError(
                f"Unknown LLM provider '{provider}'. "
                "Supported providers: openai_compatible, openai, prime, tinker"
            )

        return cls(
            base_url=base_url,
            api_key=api_key,
            model=payload["model"],
            timeout=float(payload.get("timeout", 30000.0)),
            provider=provider,  # type: ignore[arg-type]
        )


class OpenAICompatibleLLMClient:
    """Thin OpenAI-compatible chat wrapper (e.g., vLLM server)."""

    def __init__(self, *, base_url: str, api_key: str, model: str, timeout: float = 30000.0):
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.default_model = model

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=model or self.default_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
        except Exception as e:
            logger.error("OpenAI-compatible API call failed: %s", e)
            raise
        content = response.choices[0].message.content
        return content or ""


class OpenAINativeLLMClient:
    """OpenAI-native wrapper with OpenAI-specific semantics.

    - Uses `max_completion_tokens` (OpenAI docs: `max_tokens` is deprecated and
      not compatible with o-series reasoning models).
    - Maps `system` role to `developer` role for instruction messages.
    """

    def __init__(self, *, base_url: str, api_key: str, model: str, timeout: float = 30000.0):
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.default_model = model

    @staticmethod
    def _normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for message in messages:
            role = message.get("role", "user")
            if role == "system":
                role = "developer"
            normalized.append({"role": role, "content": message.get("content", "")})
        return normalized

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=model or self.default_model,
                messages=self._normalize_messages(messages),
                # temperature=temperature, # Not supported by OpenAI-native API
                max_completion_tokens=max_tokens,
                extra_body=extra_body,
            )
        except Exception as e:
            logger.error("OpenAI-native API call failed: %s", e)
            raise
        content = response.choices[0].message.content
        return content or ""


class LLMClient(OpenAICompatibleLLMClient):
    """Factory facade for provider-specific LLM clients."""

    @classmethod
    def from_config(
        cls,
        config: LLMConfig | dict[str, Any],
    ) -> OpenAICompatibleLLMClient | OpenAINativeLLMClient:
        if isinstance(config, dict):
            config = LLMConfig.from_dict(config)
        if config.provider == "tinker":
            from decomposer.common.tinker_client import TinkerLLMClient
            return TinkerLLMClient(
                model=config.model,
                base_url=config.base_url or None,
                timeout=config.timeout / 1000.0,
            )
        if config.provider == "openai":
            return OpenAINativeLLMClient(
                base_url=config.base_url,
                api_key=config.api_key,
                model=config.model,
                timeout=config.timeout,
            )
        # prime and openai_compatible both use OpenAI-compatible API
        return OpenAICompatibleLLMClient(
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
            timeout=config.timeout,
        )
