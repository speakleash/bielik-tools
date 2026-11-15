/** Streaming: print the answer as it arrives. */
import { collectStream, makeClient } from "./client.ts";

const { client, model } = makeClient();

const stream = await client.chat.completions.create({
  model,
  messages: [{ role: "user", content: "Jadę na jednodniową wycieczkę do Końskich. Co warto zobaczyć? Podaj 3 punkty." }],
  temperature: 0.2,
  max_tokens: 400,
  stream: true,
});

process.stdout.write("assistant: ");
const result = await collectStream(stream, { onContent: (text) => process.stdout.write(text) });
process.stdout.write("\n");
console.log(`finish_reason: ${result.finishReason}`);
