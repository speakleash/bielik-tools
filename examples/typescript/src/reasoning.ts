/**
 * v3.1 reasoning: the same question with thinking off and on.
 *
 * Usage: npm run reasoning -- [effort]   (none|low|medium|high|max, default medium)
 */
import { collectStream, isReasoningEffort, makeClient, withReasoning } from "./client.ts";

const effortArg = process.argv[2] ?? "medium";
if (!isReasoningEffort(effortArg)) {
  console.error(`unknown effort "${effortArg}"; use none|low|medium|high|max`);
  process.exit(2);
}

const { client, model } = makeClient();
const question = "Ile jest liczb pierwszych mniejszych od 50? Podaj samą liczbę i krótkie uzasadnienie.";

for (const effort of ["none", effortArg] as const) {
  console.log(`\n--- reasoning_effort=${effort} ---`);
  const stream = await client.chat.completions.create(
    withReasoning(
      {
        model,
        messages: [{ role: "user", content: question }],
        // Thinking needs room: max_tokens covers reasoning plus the answer.
        max_tokens: effort === "none" ? 400 : 4096,
        temperature: effort === "none" ? 0.2 : 0.6,
        stream: true,
      },
      effort,
    ),
  );
  let thinking = false;
  const result = await collectStream(stream, {
    onReasoning: (text) => {
      if (!thinking) process.stdout.write("[thinking]\n");
      thinking = true;
      process.stdout.write(text);
    },
    onContent: (text) => {
      if (thinking) process.stdout.write("\n[/thinking]\n");
      thinking = false;
      process.stdout.write(text);
    },
  });
  process.stdout.write("\n");
  console.log(
    `finish_reason=${result.finishReason} reasoning_chars=${result.reasoning.length} answer_chars=${result.content.length}`,
  );
  if (result.reasoning && !result.content) {
    console.warn("warning: reasoning but no answer; raise max_tokens");
  }
  if (effort !== "none" && !result.reasoning && /<think>/.test(result.content)) {
    console.warn("note: <think> came back inside content; the server has no reasoning parser for Bielik");
  }
}
