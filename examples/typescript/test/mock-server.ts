/** A fake OpenAI-compatible server built on an injected fetch. */
export interface Recorded {
  url: string;
  body: Record<string, unknown>;
}

type Reply = { json: unknown } | { sse: unknown[] };

export function mockFetch(replies: Reply[]) {
  const requests: Recorded[] = [];
  const queue = [...replies];
  const fetchImpl = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    requests.push({ url, body: JSON.parse(String(init?.body ?? "{}")) });
    const reply = queue.shift();
    if (!reply) throw new Error(`unexpected request to ${url}`);
    if ("json" in reply) {
      return new Response(JSON.stringify(reply.json), { headers: { "content-type": "application/json" } });
    }
    const body = reply.sse.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("") + "data: [DONE]\n\n";
    return new Response(body, { headers: { "content-type": "text/event-stream" } });
  }) as typeof fetch;
  return { fetch: fetchImpl, requests };
}

export function completion(message: Record<string, unknown>, finish_reason = "stop") {
  return {
    id: "cmpl-test",
    object: "chat.completion",
    created: 0,
    model: "bielik-test",
    choices: [{ index: 0, message: { role: "assistant", content: null, ...message }, finish_reason }],
  };
}

export function chunk(delta: Record<string, unknown>, finish_reason: string | null = null) {
  return {
    id: "cmpl-test",
    object: "chat.completion.chunk",
    created: 0,
    model: "bielik-test",
    choices: [{ index: 0, delta, finish_reason }],
  };
}
