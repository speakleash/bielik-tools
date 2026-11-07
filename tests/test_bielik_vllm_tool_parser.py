"""Unit tests for tools/bielik_vllm_tool_parser.py (vLLM >= 0.24)."""
import json
import re

import pytest

from conftest import FakeTokenizer, collect_stream, stream_chunks


def tool_call(name, arguments, sep=""):
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"<tool_call>{sep}{payload}{sep}</tool_call>"


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------

class TestInit:

    def test_resolves_special_token_ids(self, parser):
        assert parser.tool_call_start_token_id == 1
        assert parser.tool_call_end_token_id == 2

    def test_accepts_tools_argument(self, tool_parser_module):
        # vLLM >= 0.24 passes tools= to the constructor (issue #12).
        parser = tool_parser_module.BielikToolParser(FakeTokenizer(), tools=None)
        assert parser.tools == []

    def test_missing_special_tokens_raises(self, tool_parser_module):
        with pytest.raises(RuntimeError, match="could not locate tool call"):
            tool_parser_module.BielikToolParser(FakeTokenizer(vocab={"a": 1}))

    def test_missing_tokenizer_raises(self, tool_parser_module):
        with pytest.raises(ValueError, match="tokenizer must be passed"):
            tool_parser_module.BielikToolParser(None)

    def test_opts_out_of_json_required_path(self, tool_parser_module):
        assert tool_parser_module.BielikToolParser.supports_required_and_named is False


# --------------------------------------------------------------------------
# adjust_request
# --------------------------------------------------------------------------

class TestAdjustRequest:

    def test_auto_keeps_special_tokens(self, parser, make_request):
        request = parser.adjust_request(make_request(tool_choice="auto"))
        assert request.skip_special_tokens is False
        assert request.tool_choice == "auto"
        assert request.structured_outputs is None

    def test_none_leaves_request_untouched(self, parser, make_request):
        request = parser.adjust_request(make_request(tool_choice="none"))
        assert request.skip_special_tokens is True
        assert request.tool_choice == "none"

    def test_without_tools_leaves_request_untouched(self, parser, make_request):
        request = parser.adjust_request(make_request(tools=()))
        assert request.skip_special_tokens is True
        assert request.structured_outputs is None

    def test_required_redirects_to_auto_with_regex(self, parser, make_request):
        # Issue #9: vLLM's built-in "required" path expects a raw JSON array.
        request = parser.adjust_request(make_request(tool_choice="required"))
        assert request.tool_choice == "auto"
        assert request.skip_special_tokens is False
        assert request.response_format is None
        assert request.structured_outputs is not None
        assert request.structured_outputs.regex

    @pytest.mark.parametrize("output", [
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Kraków"}}</tool_call>',
        '<tool_call> {"name": "get_weather", "arguments": {}}</tool_call>',
        # '<' inside arguments (HTML, comparisons) must not end the call early.
        '<tool_call>{"name": "search", "arguments": {"q": "<div>a<b</div>"}}</tool_call>',
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Kraków"}}</tool_call>\n'
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Gdańsk"}}</tool_call>',
    ])
    def test_required_regex_accepts_tool_call_output(self, parser, make_request, output):
        regex = parser.adjust_request(make_request(tool_choice="required")).structured_outputs.regex
        assert re.fullmatch(regex, output)

    @pytest.mark.parametrize("output", [
        "Dzień dobry!",
        '{"name": "get_weather", "arguments": {}}',
        "<think>Myślę.</think>Odpowiedź bez narzędzia.",
        # The grammar applies after reasoning has ended; a second think block
        # there would put reasoning into content.
        '<think>Użytkownik pyta o pogodę.</think>\n'
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Gdańsk"}}</tool_call>',
    ])
    def test_required_regex_rejects_plain_answers(self, parser, make_request, output):
        regex = parser.adjust_request(make_request(tool_choice="required")).structured_outputs.regex
        assert re.fullmatch(regex, output) is None


# --------------------------------------------------------------------------
# extract_tool_calls (non-streaming)
# --------------------------------------------------------------------------

class TestExtractToolCalls:

    def test_plain_text(self, parser, make_request):
        result = parser.extract_tool_calls("Dzień dobry!", make_request())
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == "Dzień dobry!"

    def test_single_call(self, parser, make_request):
        output = tool_call("get_weather", {"city": "Kraków"})
        result = parser.extract_tool_calls(output, make_request())
        assert result.tools_called is True
        assert result.content is None
        [call] = result.tool_calls
        assert call.type == "function"
        assert call.function.name == "get_weather"
        assert json.loads(call.function.arguments) == {"city": "Kraków"}

    def test_preserves_non_ascii_arguments(self, parser, make_request):
        output = tool_call("get_weather", {"city": "Łódź"})
        [call] = parser.extract_tool_calls(output, make_request()).tool_calls
        assert "Łódź" in call.function.arguments

    def test_whitespace_around_payload(self, parser, make_request):
        # Issue #9: the model may emit a space/newline after <tool_call>.
        output = tool_call("get_weather", {"city": "Kraków"}, sep=" \n")
        result = parser.extract_tool_calls(output, make_request())
        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "get_weather"

    def test_content_before_call(self, parser, make_request):
        output = "Sprawdzę pogodę. " + tool_call("get_weather", {"city": "Kraków"})
        result = parser.extract_tool_calls(output, make_request())
        assert result.tools_called is True
        assert result.content == "Sprawdzę pogodę. "

    def test_parallel_calls(self, parser, make_request):
        output = (tool_call("get_weather", {"city": "Kraków"}) + "\n"
                  + tool_call("get_time", {"timezone": "Europe/Warsaw"}))
        result = parser.extract_tool_calls(output, make_request())
        assert [c.function.name for c in result.tool_calls] == ["get_weather", "get_time"]
        assert len({c.id for c in result.tool_calls}) == 2

    def test_unterminated_call(self, parser, make_request):
        # Generation stopped (e.g. on stop token) before </tool_call>.
        output = '<tool_call>{"name": "get_weather", "arguments": {"city": "Kraków"}}'
        result = parser.extract_tool_calls(output, make_request())
        assert result.tools_called is True
        assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Kraków"}

    def test_nested_arguments(self, parser, make_request):
        args = {"query": {"filters": [{"field": "a", "op": "=", "value": 1}]}, "limit": 10}
        [call] = parser.extract_tool_calls(tool_call("search", args), make_request()).tool_calls
        assert json.loads(call.function.arguments) == args

    def test_empty_arguments(self, parser, make_request):
        [call] = parser.extract_tool_calls(tool_call("get_db", {}), make_request()).tool_calls
        assert json.loads(call.function.arguments) == {}

    @pytest.mark.parametrize("output", [
        "<tool_call>not json</tool_call>",
        '<tool_call>{"name": "get_weather", "arguments": {"city": </tool_call>',
        '<tool_call>{"arguments": {}}</tool_call>',
    ])
    def test_malformed_call_falls_back_to_content(self, parser, make_request, output):
        result = parser.extract_tool_calls(output, make_request())
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == output


# --------------------------------------------------------------------------
# extract_tool_calls_streaming
# --------------------------------------------------------------------------

class TestExtractToolCallsStreaming:

    def test_plain_text_passthrough(self, parser, make_request):
        deltas = stream_chunks(parser, make_request(), ["Dzień", " dobry", "!"])
        content, calls = collect_stream(deltas)
        assert content == "Dzień dobry!"
        assert calls == {}

    @pytest.mark.parametrize("chunks", [
        pytest.param(
            ["<tool_call>", '{"name": "get_weather", ', '"arguments": {"city": ',
             '"Kra', 'ków"}}', "</tool_call>"],
            id="coarse"),
        pytest.param(
            ["<tool_call>", '{"name"', ': "get_weather"', ', "arguments"',
             ': {"city"', ': "Kra', 'ków"', "}}", "</tool_call>"],
            id="fine"),
        pytest.param(
            # Token split produced by the Bielik v3 tokenizer: closing
            # braces arrive as separate tokens.
            ["<tool_call>", " {", '"', "na", "me", '":', ' "', "get", "_", "we",
             "ather", '",', ' "', "ar", "gu", "ments", '":', " {", '"', "city",
             '":', ' "', "K", "raków", '"', "}", "}", "</tool_call>"],
            id="bielik-tokens"),
    ])
    def test_single_call(self, parser, make_request, chunks):
        content, calls = collect_stream(stream_chunks(parser, make_request(), chunks))
        assert content == ""
        assert list(calls) == [0]
        assert calls[0]["id"].startswith("chatcmpl-tool-")
        assert calls[0]["name"] == "get_weather"
        assert json.loads(calls[0]["arguments"]) == {"city": "Kraków"}

    def test_name_sent_once(self, parser, make_request):
        chunks = ["<tool_call>", '{"name": "get_weather", ', '"arguments": {"city": ',
                  '"Kraków"}}', "</tool_call>"]
        deltas = stream_chunks(parser, make_request(), chunks)
        names = [
            tc.function.name
            for d in deltas if d is not None
            for tc in d.tool_calls or []
            if tc.function is not None and tc.function.name
        ]
        assert names == ["get_weather"]

    def test_state_after_stream(self, parser, make_request):
        chunks = ["<tool_call>", '{"name": "get_weather", ', '"arguments": {"city": ',
                  '"Kra', 'ków"}}', "</tool_call>"]
        stream_chunks(parser, make_request(), chunks)
        assert parser.current_tool_id == 0
        assert parser.prev_tool_call_arr == [
            {"name": "get_weather", "arguments": {"city": "Kraków"}}]
        assert json.loads(parser.streamed_args_for_tool[0]) == {"city": "Kraków"}

    def test_text_then_call(self, parser, make_request):
        chunks = ["Sprawdzę", " pogodę.", "<tool_call>", '{"name": "get_weather", ',
                  '"arguments": {"city": ', '"Kra', 'ków"}}', "</tool_call>"]
        content, calls = collect_stream(stream_chunks(parser, make_request(), chunks))
        assert content == "Sprawdzę pogodę."
        assert calls[0]["name"] == "get_weather"
        assert json.loads(calls[0]["arguments"]) == {"city": "Kraków"}
