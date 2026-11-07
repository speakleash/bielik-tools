# Native LangGraph tool loops

```bash
python scripts/probe_langgraph.py --model bielik-test --family v31 \
  --output /tmp/langgraph.json
```

Install optional pinned requirements in an isolated environment. The probe
uses native StateGraph, conditional tool routing and ToolNode, not a manually
simulated agent loop. Eight scenarios run with thinking off/on and full/
streamed model output: plain, required/single, parallel, named, none, retry,
JSON Schema and conversation history. Only explicit synthetic tools execute.
Named/required choice applies to the initial generation; following turns
use auto so the agent can finish after receiving tool results.

Stock ChatOpenAI deliberately drops nonstandard reasoning fields. The
test-only ChatBielik subclass preserves both reasoning aliases in full and
streamed responses and replays reasoning in assistant history. Its private
hooks are covered by offline tests and tied to the pinned package version;
this makes no compatibility claim for arbitrary LangChain versions or the
Responses API. The optional adapter tests skip when dependencies are absent.
