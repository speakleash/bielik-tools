#!/usr/bin/env python3
"""Render-check the Jinja chat templates in ``tools/``.

Each template is rendered for a handful of representative conversations
using the same Jinja setup as ``transformers`` (an
``ImmutableSandboxedEnvironment`` with ``trim_blocks``, ``lstrip_blocks``
and the ``loopcontrols`` extension, plus the ``raise_exception`` and
``strftime_now`` globals and a JSON ``tojson`` filter).

Usage::

    python scripts/check_templates.py [TEMPLATE ...]

Without arguments every ``tools/*.jinja`` file is checked. The script exits
with a nonzero status if any template fails to compile or render, or if a
rendered prompt does not contain the expected markers.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import jinja2
from jinja2 import nodes
from jinja2.ext import Extension, loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_GLOB = "tools/*.jinja"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

SIMPLE_CHAT = [
    {"role": "system", "content": "Jesteś pomocnym asystentem."},
    {"role": "user", "content": "Cześć! Jak się masz?"},
]

TOOL_CHAT = [
    {"role": "user", "content": "Jaka jest pogoda w Krakowie?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": {"city": "Kraków"},
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": 12}'},
    {"role": "assistant", "content": "W Krakowie jest 12°C."},
    {"role": "user", "content": "A w Gdańsku?"},
]

# (name, render kwargs, substrings that must appear, substrings that must not)
ADVANCED_CASES = [
    (
        "plain chat",
        {"messages": SIMPLE_CHAT},
        ["<|im_start|>system\n", "<|im_start|>user\n", "<|im_start|>assistant\n"],
        ["<tool_call>", "<think>"],
    ),
    (
        "with tools",
        {"messages": TOOL_CHAT, "tools": [WEATHER_TOOL]},
        ['"name": "get_weather"', "<tool_call>", "<|im_start|>tool\n"],
        [],
    ),
    (
        "with tools, tool_choice=required",
        {"messages": TOOL_CHAT, "tools": [WEATHER_TOOL], "tool_choice": "required"},
        ["You MUST invoke at least one tool"],
        [],
    ),
    (
        "enable_thinking=true",
        {"messages": SIMPLE_CHAT, "enable_thinking": True},
        ["<think>"],
        ["<think>\n\n</think>"],
    ),
    (
        "enable_thinking=false",
        {"messages": SIMPLE_CHAT, "enable_thinking": False},
        ["<think>\n\n</think>"],
        [],
    ),
    (
        "with tools, enable_thinking=true",
        {"messages": TOOL_CHAT, "tools": [WEATHER_TOOL], "enable_thinking": True},
        ["<think>", '"name": "get_weather"'],
        [],
    ),
    (
        "with tools, enable_thinking=false",
        {"messages": TOOL_CHAT, "tools": [WEATHER_TOOL], "enable_thinking": False},
        ["<think>\n\n</think>", '"name": "get_weather"'],
        [],
    ),
]


V32_CASES = [
    (
        "plain chat (reasoning_effort defaults to none)",
        {"messages": SIMPLE_CHAT},
        ["<|im_start|>system\n", "<|im_start|>user\n", "<|im_start|>assistant\n<think></think>"],
        ["<tool_call>"],
    ),
    (
        "with tools",
        {"messages": TOOL_CHAT, "tools": [WEATHER_TOOL]},
        ['"name": "get_weather"', "<tool_call>", "<|im_start|>tool\n"],
        [],
    ),
    (
        "with tools, tool_choice=required",
        {"messages": TOOL_CHAT, "tools": [WEATHER_TOOL], "tool_choice": "required"},
        ["You MUST invoke at least one tool"],
        [],
    ),
    (
        "reasoning_effort=high",
        {"messages": SIMPLE_CHAT, "reasoning_effort": "high"},
        ["Current reasoning effort: high", "<|im_start|>assistant\n<think>"],
        ["<|im_start|>assistant\n<think></think>"],
    ),
    (
        "with tools, reasoning_effort=medium",
        {"messages": TOOL_CHAT, "tools": [WEATHER_TOOL], "reasoning_effort": "medium"},
        ["Current reasoning effort: medium", '"name": "get_weather"'],
        [],
    ),
]


CASES_BY_FILE = {
    "bielik_advanced_chat_template.jinja": ADVANCED_CASES,
    "bielik_v3.2_chat_template.jinja": V32_CASES,
}


def cases_for(path, source):
    """Cases for a known template file; otherwise guess from its switches."""
    if path.name in CASES_BY_FILE:
        return CASES_BY_FILE[path.name]
    # v3.1/v3.2 templates validate reasoning_effort; the advanced one does not.
    return V32_CASES if "reasoning_effort must be one of" in source else ADVANCED_CASES


class GenerationExtension(Extension):
    """Accept ``{% generation %}...{% endgeneration %}`` like transformers.

    transformers uses the block to mark assistant tokens for training masks;
    for rendering it is transparent.
    """

    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return nodes.Scope(body, lineno=lineno)


class TemplateError(Exception):
    """Raised by the ``raise_exception`` template global."""


def _raise_exception(message):
    raise TemplateError(message)


def _tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    # Mirrors transformers: no HTML escaping, non-ASCII kept as-is.
    return json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )


def _strftime_now(fmt):
    return datetime.now().strftime(fmt)


def make_environment():
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[loopcontrols, GenerationExtension],
    )
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise_exception
    env.globals["strftime_now"] = _strftime_now
    return env


def check_template(env, path):
    """Return a list of error messages for the template at ``path``."""
    source = path.read_text(encoding="utf-8")
    try:
        template = env.from_string(source)
    except jinja2.TemplateError as exc:
        return [f"failed to compile: {exc!r}"]

    errors = []
    for name, kwargs, must_contain, must_not_contain in cases_for(path, source):
        context = {"bos_token": "<s>", "add_generation_prompt": True, **kwargs}
        try:
            rendered = template.render(**context)
        except Exception as exc:  # noqa: BLE001 - report any render failure
            errors.append(f"[{name}] render failed: {exc!r}")
            continue
        for marker in must_contain:
            if marker not in rendered:
                errors.append(f"[{name}] missing expected text {marker!r}")
        for marker in must_not_contain:
            if marker in rendered:
                errors.append(f"[{name}] unexpected text {marker!r}")

    # An unsupported role must be rejected via raise_exception.
    try:
        template.render(
            messages=[{"role": "narrator", "content": "x"}],
            bos_token="<s>",
            add_generation_prompt=True,
        )
    except TemplateError:
        pass
    except Exception as exc:  # noqa: BLE001
        errors.append(f"[invalid role] unexpected error: {exc!r}")
    else:
        errors.append("[invalid role] template did not raise for an unknown role")

    return errors


def main(argv):
    paths = [Path(p) for p in argv] or sorted(REPO_ROOT.glob(TEMPLATE_GLOB))
    if not paths:
        print(f"No templates found matching {TEMPLATE_GLOB}", file=sys.stderr)
        return 1

    env = make_environment()
    failed = False
    for path in paths:
        errors = check_template(env, path)
        if errors:
            failed = True
            print(f"FAIL {path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"ok   {path} ({len(cases_for(path, path.read_text(encoding='utf-8'))) + 1} cases)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
