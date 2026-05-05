from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TinkerLLMClient:
    """LLMClient-compatible wrapper around Tinker's sampling client.

    Satisfies the same ``chat()`` interface as OpenAICompatibleLLMClient so it
    works as a drop-in for any LLM role (teacher, student, verifier).
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: Optional[str] = None,
        timeout: float = 300.0,
    ):
        import tinker
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        service_client = tinker.ServiceClient(base_url=base_url)
        self.sampling_client = service_client.create_sampling_client(base_model=model)

        tokenizer_id = model
        self.tokenizer = get_tokenizer(tokenizer_id)
        renderer_name = model_info.get_recommended_renderer_name(model)
        self.renderer = renderers.get_renderer(renderer_name, self.tokenizer)
        self.default_model = model
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        """Sample one completion."""
        from tinker import types

        model_input = self.renderer.build_generation_prompt(messages)
        sampling_params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=self.renderer.get_stop_sequences(),
        )
        last_err: Exception | None = None
        for attempt in range(3):
            future = self.sampling_client.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params,
            )
            try:
                result = future.result(timeout=self.timeout)
                seq = result.sequences[0]
                parsed_message, _ = self.renderer.parse_response(seq.tokens)
                return parsed_message["content"]
            except TimeoutError as e:
                last_err = e
                logger.warning(
                    "Tinker sample timed out after %.0fs (attempt %d/3)",
                    self.timeout, attempt + 1,
                )
            except Exception as e:
                last_err = e
                logger.warning(
                    "Tinker sample failed (attempt %d/3): %s", attempt + 1, e,
                )
        raise TimeoutError(
            f"Tinker sample failed after 3 attempts: {last_err}"
        ) from last_err
