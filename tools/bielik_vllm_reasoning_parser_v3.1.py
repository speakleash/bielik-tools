from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser
from vllm.tokenizers import TokenizerLike


# Registered as "bielik" so --reasoning-parser bielik works when this file is
# loaded via --reasoning-parser-plugin. Do not load this plugin and the legacy
# bielik_vllm_reasoning_parser.py in the same process (same registration name).
@ReasoningParserManager.register_module("bielik")
class BielikReasoningParser(ReasoningParser):
    """
    Reasoning parser for Bielik models with the v3.1 chat template.

    Delegates to DeepSeekR1ReasoningParser when thinking is enabled, and to
    IdentityReasoningParser otherwise — so that the entire model output is
    returned as content, not reasoning.

    Thinking is enabled when ``chat_template_kwargs`` has
    ``reasoning_effort`` set to anything other than ``"none"``. If
    ``reasoning_effort`` is absent, falls back to the legacy
    ``enable_thinking=True`` flag (advanced template).
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs") or {}
        effort = chat_kwargs.get("reasoning_effort")
        if effort is not None:
            enable_thinking = str(effort).lower() != "none"
        else:
            enable_thinking = bool(chat_kwargs.get("enable_thinking", False))

        if enable_thinking:
            self._parser = DeepSeekR1ReasoningParser(tokenizer, *args, **kwargs)
        else:
            self._parser = IdentityReasoningParser(tokenizer, *args, **kwargs)

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return self._parser.is_reasoning_end(input_ids)

    def is_reasoning_end_streaming(
        self, input_ids: list[int], delta_ids: list[int]
    ) -> bool:
        return self._parser.is_reasoning_end_streaming(input_ids, delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return self._parser.extract_content_ids(input_ids)

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        return self._parser.extract_reasoning(model_output, request)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        return self._parser.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
