/**
 * A minimal tool loop: send tools, run the calls the model asks for, send the
 * results back, repeat until the model answers in plain text.
 */
import type OpenAI from "openai";
import type {
  ChatCompletionFunctionTool,
  ChatCompletionMessageParam,
} from "openai/resources/chat/completions";
import { parseToolArguments } from "./client.ts";

export interface LocalTool {
  definition: ChatCompletionFunctionTool;
  run: (args: Record<string, unknown>) => unknown | Promise<unknown>;
}

export interface ToolLoopOptions {
  client: OpenAI;
  model: string;
  messages: ChatCompletionMessageParam[];
  tools: LocalTool[];
  maxRounds?: number;
  temperature?: number;
  maxTokens?: number;
  onToolCall?: (name: string, args: Record<string, unknown>, result: unknown) => void;
}

export interface ToolLoopResult {
  answer: string;
  rounds: number;
  toolCalls: { name: string; args: Record<string, unknown>; result: unknown }[];
  messages: ChatCompletionMessageParam[];
}

export async function runToolLoop(options: ToolLoopOptions): Promise<ToolLoopResult> {
  const { client, model, tools, maxRounds = 5 } = options;
  const messages = [...options.messages];
  const byName = new Map(tools.map((tool) => [tool.definition.function.name, tool]));
  const executed: ToolLoopResult["toolCalls"] = [];

  for (let round = 1; round <= maxRounds; round++) {
    const response = await client.chat.completions.create({
      model,
      messages,
      tools: tools.map((tool) => tool.definition),
      tool_choice: "auto",
      temperature: options.temperature ?? 0.2,
      max_tokens: options.maxTokens ?? 800,
    });
    const message = response.choices[0]?.message;
    if (!message) throw new Error("server returned no choices");

    const calls = (message.tool_calls ?? []).filter((call) => call.type === "function");
    if (calls.length === 0) {
      return { answer: message.content ?? "", rounds: round, toolCalls: executed, messages };
    }

    messages.push({ role: "assistant", content: message.content ?? null, tool_calls: calls });
    for (const call of calls) {
      const tool = byName.get(call.function.name);
      let result: unknown;
      let args: Record<string, unknown> = {};
      try {
        if (!tool) throw new Error(`unknown tool: ${call.function.name}`);
        // Tool arguments are model output: validate before acting on them.
        args = parseToolArguments(call.function.arguments);
        result = await tool.run(args);
      } catch (error) {
        result = { error: error instanceof Error ? error.message : String(error) };
      }
      executed.push({ name: call.function.name, args, result });
      options.onToolCall?.(call.function.name, args, result);
      messages.push({ role: "tool", tool_call_id: call.id, content: JSON.stringify(result) });
    }
  }
  throw new Error(`no final answer after ${maxRounds} rounds of tool calls`);
}
