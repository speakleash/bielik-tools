"""Apply narrowly scoped backports to vLLM 0.14.0, 0.15.0 or 0.20.0.

Run this script with the Python interpreter of the target vLLM environment.
Default: inspect only. Pass --apply to patch, then restart the server.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import stat
import tempfile

EFFORT_OLD = """                chat_template_kwargs = request.chat_template_kwargs or {}
                chat_template_kwargs.update(reasoning_effort=request.reasoning_effort)"""
EFFORT_NEW = """                chat_template_kwargs = request.chat_template_kwargs or {}
                if request.reasoning_effort is not None:
                    chat_template_kwargs["reasoning_effort"] = request.reasoning_effort"""

HANDOFF_OLD = """            delta_message = self.extract_tool_calls_streaming(
                previous_text=state.previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=state.previous_token_ids,
                current_token_ids=current_token_ids,
                delta_token_ids=delta_token_ids,
                request=request,  # type: ignore[arg-type]
            )"""
HANDOFF_NEW = """            reasoning_message = delta_message
            delta_message = self.extract_tool_calls_streaming(
                previous_text=state.previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=state.previous_token_ids,
                current_token_ids=current_token_ids,
                delta_token_ids=delta_token_ids,
                request=request,  # type: ignore[arg-type]
            )
            # The handoff may contain reasoning and a tool call in one chunk.
            # Content was handed to the tool parser above; retain reasoning too.
            if reasoning_message is not None and reasoning_message.reasoning:
                if delta_message is None:
                    delta_message = reasoning_message
                else:
                    delta_message.reasoning = (
                        reasoning_message.reasoning + (delta_message.reasoning or "")
                    )"""

PATCHES = {
    "0.14.0": ("vllm/entrypoints/openai/serving_chat.py", EFFORT_OLD, EFFORT_NEW),
    "0.15.0": (
        "vllm/entrypoints/openai/chat_completion/serving.py",
        EFFORT_OLD,
        EFFORT_NEW,
    ),
    "0.20.0": ("vllm/parser/abstract_parser.py", HANDOFF_OLD, HANDOFF_NEW),
}


def transform(source: str, old: str, new: str) -> tuple[str, bool]:
    """Refuse unfamiliar/partially patched code instead of guessing."""
    if source.count(new) == 1 and old not in source.replace(new, "", 1):
        return source, False
    if source.count(old) != 1 or new in source:
        raise ValueError("Unexpected vLLM source; no files changed")
    result = source.replace(old, new, 1)
    compile(result, "patched-vllm", "exec")
    return result, True


def apply_file(path: Path, old: str, new: str, *, apply: bool) -> str:
    source = path.read_text(encoding="utf-8")
    result, changed = transform(source, old, new)
    if not changed:
        return "already applied"
    if not apply:
        return "patch needed (use --apply)"
    # uv --link-mode=symlink links individual files into its immutable cache.
    # Replace the directory entry atomically; never write through that link.
    # A symlinked parent/package would still modify the shared cache: refuse it.
    if path.parent.absolute() != path.parent.resolve():
        raise ValueError("Symlinked package directory; install vLLM in copy mode first")
    backup = path.with_name(path.name + ".bielik-compat.orig")
    if backup.exists() and backup.read_text(encoding="utf-8") != source:
        raise ValueError("Backup contains different source; no files changed")
    if not backup.exists():
        with backup.open("x", encoding="utf-8") as output:
            output.write(source)
    mode = stat.S_IMODE(path.stat().st_mode)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(result)
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return "applied (original saved as " + backup.name + ")"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply and back up the exact matching patch",
    )
    args = parser.parse_args()
    distribution = importlib.metadata.distribution("vllm")
    version = distribution.version
    if version not in PATCHES:
        print(f"vLLM {version}: no backport supplied for this version; unchanged")
        return
    relative, old, new = PATCHES[version]
    path = Path(distribution.locate_file(relative)).absolute()
    print(f"vLLM {version}: {apply_file(path, old, new, apply=args.apply)}")
    print(path)


if __name__ == "__main__":
    main()
