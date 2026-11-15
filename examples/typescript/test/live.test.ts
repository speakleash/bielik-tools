/**
 * Live smoke test against a running server. Skipped unless BIELIK_LIVE=1.
 * Uses BIELIK_BASE_URL / BIELIK_MODEL / BIELIK_API_KEY like the examples.
 */
import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { collectStream, makeClient } from "../src/client.ts";
import { runToolLoop } from "../src/tool-loop.ts";

const live = process.env.BIELIK_LIVE === "1";

describe("live server", { skip: !live && "set BIELIK_LIVE=1 to run", timeout: 600_000 }, () => {
  const { client, model } = makeClient();

  it("answers a chat request", async () => {
    const response = await client.chat.completions.create({
      model,
      messages: [{ role: "user", content: "Podaj stolicę Polski jednym słowem." }],
      temperature: 0,
      max_tokens: 20,
    });
    assert.match(response.choices[0]?.message.content ?? "", /Warszaw/i);
  });

  it("streams", async () => {
    const stream = await client.chat.completions.create({
      model,
      messages: [{ role: "user", content: "Policz od 1 do 5, oddzielając liczby przecinkami." }],
      temperature: 0,
      max_tokens: 40,
      stream: true,
    });
    const result = await collectStream(stream);
    assert.match(result.content, /1.*2.*3.*4.*5/s);
  });

  it("calls a tool and uses its result", async () => {
    const result = await runToolLoop({
      client,
      model,
      temperature: 0,
      messages: [{ role: "user", content: "Jaka jest teraz temperatura w Krakowie? Użyj narzędzia." }],
      tools: [
        {
          definition: {
            type: "function",
            function: {
              name: "get_temperature",
              description: "Zwraca aktualną temperaturę w mieście w stopniach Celsjusza.",
              parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
            },
          },
          run: () => ({ temperatureC: 17 }),
        },
      ],
    });
    assert.ok(result.toolCalls.some((call) => call.name === "get_temperature"));
    assert.match(result.answer, /17/);
  });
});
