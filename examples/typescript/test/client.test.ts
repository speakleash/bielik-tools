import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  collectStream,
  getReasoning,
  isReasoningEffort,
  loadConfig,
  makeClient,
  parseToolArguments,
  withReasoning,
} from "../src/client.ts";
import { runToolLoop, type LocalTool } from "../src/tool-loop.ts";
import { chunk, completion, mockFetch } from "./mock-server.ts";

const config = { baseURL: "http://bielik.test/v1", model: "bielik-test", apiKey: "EMPTY" };

describe("loadConfig", () => {
  it("prefers BIELIK_* over VLLM_* and falls back to defaults", () => {
    assert.deepEqual(loadConfig({ BIELIK_MODEL: "a", VLLM_MODEL: "b", VLLM_BASE_URL: "http://x/v1" }), {
      baseURL: "http://x/v1",
      model: "a",
      apiKey: "EMPTY",
    });
    assert.equal(loadConfig({}).model, "speakleash/Bielik-11B-v3.1-Instruct");
  });

  it("treats empty values as unset", () => {
    assert.equal(loadConfig({ BIELIK_MODEL: "", VLLM_MODEL: "b" }).model, "b");
  });
});

describe("reasoning helpers", () => {
  it("validates effort values", () => {
    assert.ok(isReasoningEffort("max"));
    assert.ok(!isReasoningEffort("minimal"));
  });

  it("reads reasoning and reasoning_content", () => {
    assert.equal(getReasoning({ reasoning: "a" }), "a");
    assert.equal(getReasoning({ reasoning_content: "b" }), "b");
    assert.equal(getReasoning({ content: "c" }), undefined);
    assert.equal(getReasoning(null), undefined);
  });

  it("sends reasoning_effort at top level and to the chat template", async () => {
    const server = mockFetch([{ json: completion({ content: "15", reasoning: "2, 3, 5..." }) }]);
    const { client, model } = makeClient(config, { fetch: server.fetch });
    const response = await client.chat.completions.create(
      withReasoning({ model, messages: [{ role: "user", content: "?" }] }, "high"),
    );
    const body = server.requests[0]!.body;
    assert.equal(body.reasoning_effort, "high");
    assert.deepEqual(body.chat_template_kwargs, { reasoning_effort: "high" });
    assert.ok(!("enable_thinking" in (body.chat_template_kwargs as object)));
    assert.equal(getReasoning(response.choices[0]!.message), "2, 3, 5...");
  });
});

describe("collectStream", () => {
  it("joins reasoning, content and fragmented tool calls", async () => {
    const server = mockFetch([
      {
        sse: [
          chunk({ role: "assistant", reasoning: "myślę" }),
          chunk({ reasoning_content: "...", content: "Sprawdzę." }),
          chunk({ tool_calls: [{ index: 1, id: "b", type: "function", function: { name: "get_weather", arguments: '{"city":"Gdańsk"}' } }] }),
          chunk({ tool_calls: [{ index: 0, id: "a", type: "function", function: { name: "get_weather", arguments: '{"ci' } }] }),
          chunk({ tool_calls: [{ index: 0, function: { arguments: 'ty":"Kraków"}' } }] }, "tool_calls"),
          { id: "x", object: "chat.completion.chunk", created: 0, model: "m", choices: [], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } },
        ],
      },
    ]);
    const { client, model } = makeClient(config, { fetch: server.fetch });
    const stream = await client.chat.completions.create({ model, messages: [], stream: true });
    const seen: string[] = [];
    const result = await collectStream(stream, { onContent: (t) => seen.push(t) });
    assert.equal(result.reasoning, "myślę...");
    assert.equal(result.content, "Sprawdzę.");
    assert.deepEqual(seen, ["Sprawdzę."]);
    assert.equal(result.finishReason, "tool_calls");
    assert.deepEqual(
      result.toolCalls.map((c) => [c.id, c.function.name, c.function.arguments]),
      [
        ["a", "get_weather", '{"city":"Kraków"}'],
        ["b", "get_weather", '{"city":"Gdańsk"}'],
      ],
    );
  });
});

describe("parseToolArguments", () => {
  it("accepts objects and rejects other JSON", () => {
    assert.deepEqual(parseToolArguments('{"city":"Kraków"}'), { city: "Kraków" });
    assert.deepEqual(parseToolArguments(""), {});
    assert.throws(() => parseToolArguments("[1]"));
    assert.throws(() => parseToolArguments("{"));
  });
});

describe("runToolLoop", () => {
  const weather: LocalTool = {
    definition: {
      type: "function",
      function: { name: "get_weather", parameters: { type: "object", properties: { city: { type: "string" } } } },
    },
    run: ({ city }) => ({ city, temperatureC: city === "Kraków" ? 4 : 7 }),
  };

  it("executes calls, returns results and stops on a text answer", async () => {
    const server = mockFetch([
      {
        json: completion(
          {
            tool_calls: [
              { id: "c1", type: "function", function: { name: "get_weather", arguments: '{"city":"Kraków"}' } },
              { id: "c2", type: "function", function: { name: "nope", arguments: "{}" } },
              { id: "c3", type: "function", function: { name: "get_weather", arguments: "not json" } },
            ],
          },
          "tool_calls",
        ),
      },
      { json: completion({ content: "W Krakowie są 4°C." }) },
    ]);
    const { client, model } = makeClient(config, { fetch: server.fetch });
    const result = await runToolLoop({ client, model, tools: [weather], messages: [{ role: "user", content: "Pogoda?" }] });

    assert.equal(result.answer, "W Krakowie są 4°C.");
    assert.equal(result.rounds, 2);
    assert.deepEqual(result.toolCalls[0], { name: "get_weather", args: { city: "Kraków" }, result: { city: "Kraków", temperatureC: 4 } });
    assert.match(JSON.stringify(result.toolCalls[1]!.result), /unknown tool/);
    assert.ok("error" in (result.toolCalls[2]!.result as object));

    const second = server.requests[1]!.body.messages as { role: string; tool_call_id?: string; content?: string }[];
    assert.deepEqual(second.map((m) => m.role), ["user", "assistant", "tool", "tool", "tool"]);
    assert.equal(second[2]!.tool_call_id, "c1");
    assert.deepEqual(JSON.parse(second[2]!.content!), { city: "Kraków", temperatureC: 4 });
    assert.equal((server.requests[0]!.body.tools as unknown[]).length, 1);
  });

  it("gives up after maxRounds", async () => {
    const call = completion({ tool_calls: [{ id: "c", type: "function", function: { name: "get_weather", arguments: '{"city":"Kraków"}' } }] }, "tool_calls");
    const server = mockFetch([{ json: call }, { json: call }]);
    const { client, model } = makeClient(config, { fetch: server.fetch });
    await assert.rejects(
      runToolLoop({ client, model, tools: [weather], messages: [], maxRounds: 2 }),
      /no final answer after 2 rounds/,
    );
  });
});
