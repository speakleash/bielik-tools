/**
 * Small helpers shared by the Bielik TypeScript examples.
 *
 * Everything goes through the official `openai` package, pointed at any
 * OpenAI-compatible server (vLLM, llama.cpp `llama-server`, Ollama).
 * The helpers only cover what Bielik needs on top of the plain SDK:
 * configuration from the environment, the v3.1 `reasoning_effort` switch,
 * reading the separate reasoning field, and collecting streamed deltas.
 */
import OpenAI from "openai";
import type {
  ChatCompletionChunk,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionCreateParamsStreaming,
  ChatCompletionMessageFunctionToolCall,
} from "openai/resources/chat/completions";

/** Thinking levels understood by the Bielik v3.1/v3.2 chat template. */
export const REASONING_EFFORTS = ["none", "low", "medium", "high", "max"] as const;
export type ReasoningEffort = (typeof REASONING_EFFORTS)[number];

export interface BielikConfig {
  baseURL: string;
  model: string;
  apiKey: string;
}

const DEFAULTS: BielikConfig = {
  baseURL: "http://127.0.0.1:8000/v1",
  model: "speakleash/Bielik-11B-v3.1-Instruct",
  apiKey: "EMPTY",
};

/**
 * Read the server settings. `BIELIK_*` wins; the `VLLM_*` names used by the
 * Python examples in this repository are accepted as a fallback.
 */
export function loadConfig(env: NodeJS.ProcessEnv = process.env): BielikConfig {
  const pick = (name: "BASE_URL" | "MODEL" | "API_KEY") =>
    env[`BIELIK_${name}`] || env[`VLLM_${name}`] || undefined;
  return {
    baseURL: pick("BASE_URL") ?? DEFAULTS.baseURL,
    model: pick("MODEL") ?? DEFAULTS.model,
    apiKey: pick("API_KEY") ?? DEFAULTS.apiKey,
  };
}

export interface ClientOptions {
  /** Inject a fetch implementation (used by the offline tests). */
  fetch?: typeof fetch;
  timeoutMs?: number;
}

export function makeClient(config: BielikConfig = loadConfig(), options: ClientOptions = {}) {
  const client = new OpenAI({
    baseURL: config.baseURL,
    apiKey: config.apiKey,
    timeout: options.timeoutMs ?? 300_000,
    maxRetries: 0,
    ...(options.fetch ? { fetch: options.fetch } : {}),
  });
  return { client, model: config.model };
}

export function isReasoningEffort(value: unknown): value is ReasoningEffort {
  return typeof value === "string" && (REASONING_EFFORTS as readonly string[]).includes(value);
}

/**
 * Request fields that switch Bielik v3.1 thinking.
 *
 * - `chat_template_kwargs.reasoning_effort` is read by the chat template
 *   (vLLM and llama.cpp `--jinja`). This is the one that matters.
 * - top-level `reasoning_effort` is also accepted by vLLM 0.32 and lets the
 *   Bielik reasoning parser know whether to expect a `<think>` block.
 *
 * `enable_thinking` (Bielik v3.0) is ignored by the v3.1 template, so it is
 * not sent. Servers that do not know `chat_template_kwargs` ignore it.
 */
export function reasoningParams(effort: ReasoningEffort) {
  return {
    reasoning_effort: effort,
    chat_template_kwargs: { reasoning_effort: effort },
  };
}

type Chat = ChatCompletionCreateParamsNonStreaming;
type ChatStream = ChatCompletionCreateParamsStreaming;

/** Add Bielik reasoning fields to a request without fighting the SDK types. */
export function withReasoning<T extends Chat | ChatStream>(params: T, effort: ReasoningEffort): T {
  return { ...params, ...reasoningParams(effort) } as T;
}

/**
 * The separated reasoning text, if the server returned one. vLLM uses
 * `reasoning` (older releases `reasoning_content`), llama.cpp uses
 * `reasoning_content`, Ollama uses `reasoning`.
 */
export function getReasoning(messageOrDelta: unknown): string | undefined {
  if (!messageOrDelta || typeof messageOrDelta !== "object") return undefined;
  const record = messageOrDelta as Record<string, unknown>;
  for (const key of ["reasoning", "reasoning_content"]) {
    const value = record[key];
    if (typeof value === "string" && value.length > 0) return value;
  }
  return undefined;
}

export interface CollectedStream {
  content: string;
  reasoning: string;
  toolCalls: ChatCompletionMessageFunctionToolCall[];
  finishReason: string | null;
}

export interface StreamHandlers {
  onContent?: (text: string) => void;
  onReasoning?: (text: string) => void;
}

/**
 * Drain a streamed chat completion: concatenate content and reasoning and
 * merge tool-call fragments by `index` (a server may send a call whole or in
 * pieces, see the parser notes in the main README).
 */
export async function collectStream(
  stream: AsyncIterable<ChatCompletionChunk>,
  handlers: StreamHandlers = {},
): Promise<CollectedStream> {
  const result: CollectedStream = { content: "", reasoning: "", toolCalls: [], finishReason: null };
  const calls = new Map<number, ChatCompletionMessageFunctionToolCall>();

  for await (const chunk of stream) {
    // Usage-only chunks (stream_options.include_usage) have no choices.
    const choice = chunk.choices[0];
    if (!choice) continue;
    const delta = choice.delta;

    const reasoning = getReasoning(delta);
    if (reasoning) {
      result.reasoning += reasoning;
      handlers.onReasoning?.(reasoning);
    }
    // One delta may carry the end of reasoning and the start of the answer.
    if (delta.content) {
      result.content += delta.content;
      handlers.onContent?.(delta.content);
    }
    for (const part of delta.tool_calls ?? []) {
      const call = calls.get(part.index) ?? {
        id: "",
        type: "function" as const,
        function: { name: "", arguments: "" },
      };
      if (part.id) call.id = part.id;
      if (part.function?.name) call.function.name += part.function.name;
      if (part.function?.arguments) call.function.arguments += part.function.arguments;
      calls.set(part.index, call);
    }
    if (choice.finish_reason) result.finishReason = choice.finish_reason;
  }

  result.toolCalls = [...calls.entries()].sort(([a], [b]) => a - b).map(([, call]) => call);
  return result;
}

/** Parse tool arguments defensively: they come from the model, not from you. */
export function parseToolArguments(raw: string): Record<string, unknown> {
  if (!raw.trim()) return {};
  const value: unknown = JSON.parse(raw);
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`tool arguments must be a JSON object, got: ${raw}`);
  }
  return value as Record<string, unknown>;
}
