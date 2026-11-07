"""A named tool_choice must force exactly one call of that function."""

import json
import re

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
)

from test_streaming_tool_parser_v31 import TOOL, collect, flatten, request_data, tokenizer


def named(name):
    return ChatCompletionNamedToolChoiceParam(
        type="function", function={"name": name})


NAMED = named("calc")


@pytest.fixture
def adjusted(tokenizer, request_data):
    request_data.tool_choice = NAMED
    parser = TOOL(tokenizer, request_data.tools)
    return parser, parser.adjust_request(request_data)


def test_named_choice_becomes_constrained_auto(adjusted):
    _, request = adjusted
    assert request.tool_choice == "auto"
    assert request.skip_special_tokens is False
    assert request.structured_outputs.regex.startswith(r"\s{0,2}<tool_call>")


@pytest.mark.parametrize("arguments", [
    {}, {"x": 2}, {"query": 'a <b> "q" </tool_call> z'}, {"n": {"k": [1, None]}},
])
def test_named_regex_accepts_one_call_of_that_function(adjusted, arguments):
    regex = adjusted[1].structured_outputs.regex
    for sep in ((", ", ": "), (",", ":")):
        call = ('<tool_call>{"name"%s"calc"%s"arguments"%s%s}</tool_call>'
                % (sep[1], sep[0], sep[1], json.dumps(arguments)))
        assert re.fullmatch(regex, call), call
        assert re.fullmatch(regex, " \n" + call + "\n")


@pytest.mark.parametrize("output", [
    '<tool_call>{"name": "other", "arguments": {}}</tool_call>',
    '<tool_call>{"name": "calculator", "arguments": {}}</tool_call>',
    'Sprawdzę. <tool_call>{"name": "calc", "arguments": {}}</tool_call>',
    '<tool_call>{"name": "calc", "arguments": {}}',
    '<tool_call>{"name": "calc", "arguments": {}}</tool_call>'
    '<tool_call>{"name": "calc", "arguments": {}}</tool_call>',
    '<tool_call>{"name": "calc", "arguments": {}}</tool_call>'
    '<tool_call>{"name": "other", "arguments": {}}</tool_call>',
    'Cześć!',
])
def test_named_regex_rejects_other_output(adjusted, output):
    assert re.fullmatch(adjusted[1].structured_outputs.regex, output) is None


def test_named_regex_escapes_function_names(tokenizer, request_data):
    request_data.tool_choice = named("a.b(c)")
    parser = TOOL(tokenizer, request_data.tools)
    regex = parser.adjust_request(request_data).structured_outputs.regex
    assert re.fullmatch(regex, '<tool_call>{"name": "a.b(c)", "arguments": {}}</tool_call>')
    assert re.fullmatch(regex, '<tool_call>{"name": "aXb(c)", "arguments": {}}</tool_call>') is None


def test_named_streaming_with_ordinary_token_tags(adjusted, tokenizer):
    parser, request = adjusted
    text = '<tool_call>{"name": "calc", "arguments": {"x": 2}}</tool_call>'
    deltas = collect(parser, tokenizer, request, text, 3, split_special_tokens=True)
    assert flatten(deltas) == [('calc', {'x': 2})]
    assert not ''.join(d.content or '' for d in deltas)


def test_other_tool_choices_unchanged(tokenizer, request_data):
    for choice in ("auto", "none"):
        request_data.tool_choice = choice
        parser = TOOL(tokenizer, request_data.tools)
        assert parser.adjust_request(request_data).structured_outputs is None


def test_named_nesting_limit_is_four_levels(adjusted):
    regex = adjusted[1].structured_outputs.regex
    wrap = '<tool_call>{"name": "calc", "arguments": %s}</tool_call>'
    assert re.fullmatch(regex, wrap % '{"a":{"b":{"c":{"d":1}}}}')
    assert re.fullmatch(regex, wrap % '{"a":{"b":{"c":{"d":{"e":1}}}}}') is None


def test_named_nonstreaming_extracts_forced_output(adjusted):
    parser, request = adjusted
    text = '<tool_call>{"name": "calc", "arguments": {"x": 2}}</tool_call>'
    result = parser.extract_tool_calls(text, request)
    assert result.tools_called
    assert result.tool_calls[0].function.name == "calc"
    assert json.loads(result.tool_calls[0].function.arguments) == {"x": 2}


def test_required_still_uses_required_regex(tokenizer, request_data):
    request_data.tool_choice = "required"
    parser = TOOL(tokenizer, request_data.tools)
    regex = parser.adjust_request(request_data).structured_outputs.regex
    two = '<tool_call>{"name":"a","arguments":{}}</tool_call>' * 2
    assert re.fullmatch(regex, two)


@pytest.fixture(scope="module")
def grammar_compiler():
    xgrammar = pytest.importorskip("xgrammar")
    # A byte vocabulary keeps this test offline and independent of model files.
    vocab = [bytes([i]) for i in range(256)]
    return xgrammar.GrammarCompiler(xgrammar.TokenizerInfo(vocab))


def accepts_json_call(adjusted, grammar_compiler, arguments):
    """Check the decoder's grammar, not just Python's regex implementation."""
    import xgrammar

    text = '<tool_call>{"name":"calc","arguments":' + arguments + '}</tool_call>'
    regex = adjusted[1].structured_outputs.regex
    python_accepts = re.fullmatch(regex, text) is not None
    grammar = grammar_compiler.compile_grammar(xgrammar.Grammar.from_regex(regex))
    matcher = xgrammar.GrammarMatcher(grammar)
    decoder_accepts = matcher.accept_string(text)
    return text, python_accepts, decoder_accepts


@pytest.mark.parametrize("arguments", [
    r'{"x":01}', r'{"x":-01}', r'{"x":1.}', r'{"x":.1}',
    r'{"x":+1}', r'{"x":1e}', '{"x":١}',
    r'{"x":"\q"}', r'{"x":"\uZZZZ"}', r'{"x":"\u123"}',
    r'{"\q":1}', '{"x":"raw\nnewline"}',
    '{\v"x":1}', '{"x":\f1}', '{"x":1\u00a0}',
])
def test_named_grammar_rejects_invalid_json(adjusted, grammar_compiler, arguments):
    with pytest.raises(json.JSONDecodeError):
        json.loads(arguments)
    _, python_accepts, decoder_accepts = accepts_json_call(
        adjusted, grammar_compiler, arguments)
    assert not python_accepts
    assert not decoder_accepts


@pytest.mark.parametrize("arguments", [
    r'{"x":0,"y":-0,"z":10}',
    r'{"x":-0.125,"y":1e+20,"z":2E-3}',
    r'{"x":"quote: \" slash: \/ backslash: \\"}',
    r'{"x":"\b\f\n\r\t\u0142\uD83D\uDE00"}',
    '{"zażółć":"gęślą 😀","x":[true,false,null,{"y":1}]}',
    '{\n"x":\t1\r}',
])
def test_named_grammar_accepts_json_and_extracts(
    adjusted, grammar_compiler, arguments,
):
    text, python_accepts, decoder_accepts = accepts_json_call(
        adjusted, grammar_compiler, arguments)
    assert python_accepts
    assert decoder_accepts
    parser, request = adjusted
    result = parser.extract_tool_calls(text, request)
    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "calc"
    assert json.loads(result.tool_calls[0].function.arguments) == json.loads(arguments)


@pytest.mark.parametrize("fragment", [
    '{\v"name":"calc","arguments":{}}',
    '{"name"\f:"calc","arguments":{}}',
    '{"name":\u00a0"calc","arguments":{}}',
    '{"name":"calc"\v,"arguments":{}}',
    '{"name":"calc",\f"arguments":{}}',
    '{"name":"calc","arguments"\v:{}}',
    '{"name":"calc","arguments":\u00a0{}}',
    '{"name":"calc","arguments":{}\f}',
])
def test_named_grammar_rejects_invalid_outer_json_whitespace(
    adjusted, grammar_compiler, fragment,
):
    import xgrammar

    text = '<tool_call>' + fragment + '</tool_call>'
    with pytest.raises(json.JSONDecodeError):
        json.loads(fragment)
    regex = adjusted[1].structured_outputs.regex
    assert re.fullmatch(regex, text) is None
    grammar = grammar_compiler.compile_grammar(xgrammar.Grammar.from_regex(regex))
    assert not xgrammar.GrammarMatcher(grammar).accept_string(text)
