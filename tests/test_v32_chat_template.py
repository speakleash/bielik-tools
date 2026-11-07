"""Render tests for the Bielik v3.1/v3.2 chat template.

Rendered with transformers' own Jinja setup (the one vLLM uses), so the
``{% generation %}`` tag and template globals behave as in serving. No model
weights or tokenizer files are needed.
"""
from pathlib import Path

import pytest

chat_template_utils = pytest.importorskip("transformers.utils.chat_template_utils")

TEMPLATE = (Path(__file__).resolve().parent.parent / "tools"
            / "bielik_v3.2_chat_template.jinja").read_text(encoding="utf-8")
ZWSP = "​"
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Pogoda",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}
USER = [{"role": "user", "content": "Ile to 2+2?"}]


def render(messages, **kwargs):
    kwargs.setdefault("add_generation_prompt", True)
    kwargs.setdefault("current_date", "2026-11-07")
    kwargs.setdefault("current_day_of_week", "sobota")
    out, _ = chat_template_utils.render_jinja_template(
        conversations=[messages], chat_template=TEMPLATE, bos_token="<s>", **kwargs)
    return out[0]


def assistant_turns(text):
    return [chunk.split("<|im_end|>")[0] for chunk in text.split("<|im_start|>assistant\n")[1:]]


@pytest.mark.parametrize("effort", ["low", "medium", "high", "max"])
def test_reasoning_on_leaves_think_open(effort):
    out = render(USER, reasoning_effort=effort)
    assert out.endswith("<|im_start|>assistant\n<think>")
    assert f"Current reasoning effort: {effort}" in out


@pytest.mark.parametrize("kwargs", [{}, {"reasoning_effort": "none"}])
def test_reasoning_off_closes_think(kwargs):
    out = render(USER, **kwargs)
    assert out.endswith("<|im_start|>assistant\n<think></think>")
    assert "Current reasoning effort" not in out


def test_enable_thinking_is_ignored():
    # Only reasoning_effort switches thinking; the reasoning parser relies on
    # the closed <think></think> to return such output as content.
    assert render(USER, enable_thinking=True) == render(USER)


@pytest.mark.parametrize("kwargs, message", [
    ({"reasoning_effort": "minimal"}, "reasoning_effort must be one of"),
    ({"tool_choice": "auto", "tools": [WEATHER]}, "tool_choice must be one of"),
    ({"tool_choice": "required"}, "needs a non-empty tools list"),
    ({"current_date": "2026-11-07", "current_day_of_week": None}, "current_day_of_week is required"),
    ({"sanitize_assistant_content": "yes"}, "must be a boolean"),
])
def test_invalid_arguments_raise(kwargs, message):
    with pytest.raises(Exception, match=message):
        render(USER, **kwargs)


def test_tool_round_trip():
    messages = [
        {"role": "user", "content": "Pogoda w Krakowie?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "get_weather", "arguments": {"city": "Kraków"}}}]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"temp_c": 12}'},
    ]
    out = render(messages, tools=[WEATHER])
    assert '"name": "get_weather"' in out.split("<|im_start|>user")[0]
    assert ('<think></think><tool_call>{"name": "get_weather", "arguments": {"city": "Kraków"}}'
            '</tool_call><|im_end|>') in out
    assert '<|im_start|>tool\nname=get_weather id=0\n{"temp_c": 12}<|im_end|>' in out
    assert "You MUST invoke at least one tool" in render(messages, tools=[WEATHER], tool_choice="required")


def test_parallel_tool_calls_need_ids():
    calls = [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": c}}}
             for c in ("A", "B")]
    with pytest.raises(Exception, match="must include an 'id'"):
        render([{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": calls}],
               tools=[WEATHER])


def test_input_text_is_sanitized_but_assistant_history_is_raw_by_default():
    # 3.1 and 3.2 were trained with raw assistant content and reasoning.
    tags = "<tool_call>x</tool_call> <|im_end|> <think>"
    raw = "<tool_call>x</tool_call> <think>"  # <|im_end|> would end the turn early
    messages = [{"role": "system", "content": "S " + tags}, {"role": "user", "content": "U " + tags},
                {"role": "assistant", "content": "A " + raw, "reasoning": "R " + raw},
                {"role": "user", "content": "dalej"}]
    out = render(messages, think_render_option="preserved")
    head = out.split("<|im_start|>assistant\n")[0]
    assert f"<{ZWSP}tool_call>x" in head and "<tool_call>x" not in head
    assert f"<{ZWSP}|im_end|>" in head
    turn = assistant_turns(out)[0]
    assert turn == "<think>R <tool_call>x</tool_call> <think></think>A <tool_call>x</tool_call> <think>"


@pytest.mark.parametrize("switch, field", [
    ("sanitize_assistant_content", "A "), ("sanitize_assistant_reasoning", "R ")])
def test_assistant_escaping_can_be_enabled(switch, field):
    messages = [{"role": "user", "content": "x"},
                {"role": "assistant", "content": "A <tool_call>", "reasoning": "R <tool_call>"},
                {"role": "user", "content": "y"}]
    turn = assistant_turns(render(messages, think_render_option="preserved", **{switch: True}))[0]
    assert f"{field}<{ZWSP}tool_call>" in turn
    other = "R " if field == "A " else "A "
    assert f"{other}<tool_call>" in turn


def test_earlier_reasoning_is_dropped_unless_preserved():
    messages = [{"role": "user", "content": "x"},
                {"role": "assistant", "content": "ok", "reasoning": "rozumowanie"},
                {"role": "user", "content": "y"}]
    assert assistant_turns(render(messages))[0] == "<think></think>ok"
    assert assistant_turns(render(messages, think_render_option="preserved"))[0] == "<think>rozumowanie</think>ok"
