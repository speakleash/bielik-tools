# Truncated and malformed calls

`test_malformed_tool_outputs.py` cuts a valid call at every character
boundary and feeds each prefix to both parser paths. A complete JSON object
without `</tool_call>` is still incomplete and must never become a call.
Malformed completed calls remain data: missing/wrong names, non-object
arguments, invalid JSON escapes and non-finite numeric values are rejected.

These tests exposed gaps in non-streaming extraction, so this slice also
requires complete wrappers, shares payload validation across both paths and
preserves text between/after calls. Closing-tag text inside a valid JSON
string is treated as data, not a wrapper boundary. No tool is executed.

Unclosed blocks may remain visible as raw data in a full response; streaming
buffers them rather than inventing a completed call. Whether a syntactically
valid name/argument object matches a tool's schema remains the caller's
validation responsibility, as before.
