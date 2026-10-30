"""
This parser works with vLLM <=0.12.0. For newer vLLM versions, use bielik_vllm_tool_parser.py
"""
import json
import re
from typing import Union, Sequence


from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager
try:
    from vllm.sampling_params import StructuredOutputsParams
except ImportError:
    StructuredOutputsParams = None  # older vLLM used guided decoding params
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer
try:
    from vllm.tokenizers.mistral import MistralTokenizer
except ModuleNotFoundError as exc:
    if not exc.name.startswith("vllm.tokenizers"):
        raise
    from vllm.transformers_utils.tokenizer import MistralTokenizer
from vllm.utils import random_uuid

logger = init_logger(__name__)


@ToolParserManager.register_module("bielik")
class BielikToolParser(ToolParser):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        if isinstance(self.model_tokenizer, MistralTokenizer):
            logger.warning("Detected Mistral tokenizer when using a Bielik model")
            self.model_tokenizer = self.model_tokenizer.tokenizer

        self._stream_text_offset = 0
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = [
        ]  # map what has been streamed for each tool so far to a list

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"

        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")
        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError("Bielik Tool parser could not locate tool call start/end tokens in the tokenizer!")
      
    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if request.tools and request.tool_choice != 'none':
            # do not skip special tokens because Bielik uses the special tokens
            # to indicated the start and end of the tool calls information.
            request.skip_special_tokens = False
            if (request.tool_choice == "required" and StructuredOutputsParams is not None
                    and hasattr(request, "structured_outputs")):
                request.tool_choice = "auto"
                request.response_format = None
                request.structured_outputs = StructuredOutputsParams(
                    regex=r"(<think>[^<]*</think>\s*)?(<tool_call>[^<]+</tool_call>)"
                )
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:

        # sanity check; avoid unnecessary processing
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)
        else:
            try:
                # there are two possible captures - between tags, or between a
                # tag and end-of-string so the result of
                # findall is an array of tuples where one is a function call and
                # the other is None
                function_call_tuples = self.tool_call_regex.findall(model_output)

                # load the JSON, and then use it to build the Function and Tool Call
                raw_function_calls = [
                    json.loads(match[0] if match[0] else match[1])
                    for match in function_call_tuples
                ]
                tool_calls = [
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=function_call["name"],
                            # function call args are JSON but as a string
                            arguments=json.dumps(function_call["arguments"],
                                                 ensure_ascii=False)))
                    for function_call in raw_function_calls
                ]

                content = model_output[:model_output.find(self.tool_call_start_token)]
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None)
            except Exception:
                logger.exception("Error in extracting tool call from response.")
                return ExtractedToolCallInformation(tools_called=False,
                                                    tool_calls=[],
                                                    content=model_output)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:

        # current_text is cumulative. Consume each text range/closed call once;
        # never concatenate delta_text again or infer state from balanced counts.
        if self.tool_call_start_token_id not in current_token_ids:
            self._stream_text_offset = len(current_text)
            return DeltaMessage(content=delta_text) if delta_text else None

        cursor = self._stream_text_offset
        content = []
        calls = []
        while cursor < len(current_text):
            start = current_text.find(self.tool_call_start_token, cursor)
            if start < 0:
                content.append(current_text[cursor:])
                cursor = len(current_text)
                break
            content.append(current_text[cursor:start])
            payload_start = start + len(self.tool_call_start_token)
            end = current_text.find(self.tool_call_end_token, payload_start)
            if end < 0:
                cursor = start  # wait for the rest of this call; do not emit partial JSON
                break
            next_cursor = end + len(self.tool_call_end_token)
            try:
                call = json.loads(current_text[payload_start:end])
                if (not isinstance(call, dict) or not isinstance(call.get("name"), str)
                        or not call["name"] or not isinstance(call.get("arguments"), dict)):
                    raise ValueError("Expected a tool name and an arguments object")
                arguments = json.dumps(call["arguments"], ensure_ascii=False)
            except (ValueError, TypeError):
                # Preserve malformed completed blocks as text; never execute them.
                logger.warning("Invalid completed tool call in streaming output")
                content.append(current_text[start:next_cursor])
                cursor = next_cursor
                continue

            self.current_tool_id += 1
            self.current_tool_name_sent = True
            self.prev_tool_call_arr.append(call)
            self.streamed_args_for_tool.append(arguments)
            calls.append(DeltaToolCall(
                index=self.current_tool_id,
                type="function",
                id=f"chatcmpl-tool-{random_uuid()}",
                function=DeltaFunctionCall(name=call["name"], arguments=arguments),
            ))
            cursor = next_cursor

        self._stream_text_offset = cursor
        text = "".join(content)
        if not text and not calls:
            return None
        return DeltaMessage(content=text or None, tool_calls=calls)
