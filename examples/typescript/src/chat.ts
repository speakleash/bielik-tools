/** Basic chat: one request, one answer. */
import { makeClient } from "./client.ts";

const { client, model } = makeClient();

const response = await client.chat.completions.create({
  model,
  messages: [
    { role: "system", content: "Odpowiadaj krótko i po polsku." },
    { role: "user", content: "Wymyśl krótkie motywujące zdanie na dziś." },
  ],
  temperature: 0.2,
  max_tokens: 200,
});

const choice = response.choices[0];
console.log(`model: ${response.model}`);
console.log(`finish_reason: ${choice?.finish_reason}`);
console.log(`assistant: ${choice?.message.content?.trim()}`);
if (response.usage) {
  console.log(`tokens: prompt=${response.usage.prompt_tokens} completion=${response.usage.completion_tokens}`);
}
