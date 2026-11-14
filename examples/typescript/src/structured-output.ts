/**
 * Structured output with `response_format: json_schema`. The server
 * constrains decoding to the schema; the result is still validated here.
 */
import OpenAI from "openai";
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import { makeClient } from "./client.ts";

const schema = {
  type: "object",
  properties: {
    brand: { type: "string" },
    model: { type: "string" },
    car_type: { type: "string", enum: ["sedan", "SUV", "kombi", "hatchback", "coupe"] },
    reason: { type: "string" },
  },
  required: ["brand", "model", "car_type", "reason"],
  additionalProperties: false,
} as const;

interface Car {
  brand: string;
  model: string;
  car_type: (typeof schema.properties.car_type.enum)[number];
  reason: string;
}

/** Models sometimes wrap JSON in a ```json fence when it is not enforced. */
function stripCodeFence(text: string): string {
  const match = text.trim().match(/^```(?:json)?\s*([\s\S]*?)\s*```$/);
  return match ? match[1]! : text.trim();
}

function parseCar(text: string): Car {
  const value = JSON.parse(text) as Partial<Car>;
  for (const key of schema.required) {
    if (typeof value[key] !== "string") throw new Error(`missing or invalid field: ${key}`);
  }
  if (!schema.properties.car_type.enum.includes(value.car_type as Car["car_type"])) {
    throw new Error(`car_type outside the enum: ${value.car_type}`);
  }
  return value as Car;
}

const { client, model } = makeClient();
const messages: ChatCompletionMessageParam[] = [
  {
    role: "user",
    content: "Jaki samochód poleciłbyś czteroosobowej rodzinie w Polsce? Odpowiedz w JSON: marka, model, typ nadwozia i krótkie uzasadnienie.",
  },
];

let raw: string;
try {
  const response = await client.chat.completions.create({
    model,
    messages,
    response_format: { type: "json_schema", json_schema: { name: "Car", schema, strict: true } },
    temperature: 0.2,
    max_tokens: 400,
  });
  const choice = response.choices[0];
  if (choice?.finish_reason === "length") console.warn("warning: hit max_tokens, JSON may be cut off");
  raw = choice?.message.content ?? "";
  console.log("mode: response_format json_schema (server-side constrained decoding)");
} catch (error) {
  // llama.cpp (and Ollama, which embeds it) cannot build the grammar for the
  // Bielik v3.1 chat template yet and answers HTTP 400 "Failed to initialize
  // samplers". Fall back to asking for JSON in the prompt and validating it.
  if (!(error instanceof OpenAI.BadRequestError) || !/sampler|grammar/i.test(error.message)) throw error;
  console.warn(`note: server rejected response_format (${error.message}); falling back to prompt-only JSON`);
  const response = await client.chat.completions.create({
    model,
    messages: [
      {
        role: "system",
        content: `Odpowiadaj wyłącznie poprawnym obiektem JSON zgodnym z tym schematem, bez komentarzy i bez bloków kodu: ${JSON.stringify(schema)}`,
      },
      ...messages,
    ],
    temperature: 0.2,
    max_tokens: 400,
  });
  raw = stripCodeFence(response.choices[0]?.message.content ?? "");
  console.log("mode: prompt-only JSON (not constrained, validated below)");
}

console.log(`raw: ${raw}`);
console.log("parsed:", parseCar(raw));
