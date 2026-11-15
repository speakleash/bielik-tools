/**
 * Tool calling with a real loop. The weather data is fixed so the run is
 * reproducible; swap `run` for a real API call.
 */
import { makeClient } from "./client.ts";
import { runToolLoop, type LocalTool } from "./tool-loop.ts";

const WEATHER: Record<string, { temperatureC: number; conditions: string }> = {
  krakow: { temperatureC: 4, conditions: "mżawka" },
  gdansk: { temperatureC: 7, conditions: "pochmurno, silny wiatr" },
  warszawa: { temperatureC: 5, conditions: "zachmurzenie umiarkowane" },
};

const normalise = (city: string) =>
  city.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/ł/g, "l").trim();

const getWeather: LocalTool = {
  definition: {
    type: "function",
    function: {
      name: "get_weather",
      description: "Zwraca aktualną pogodę dla podanego miasta w Polsce.",
      parameters: {
        type: "object",
        properties: { city: { type: "string", description: "Nazwa miasta, np. Kraków" } },
        required: ["city"],
        additionalProperties: false,
      },
    },
  },
  run: ({ city }) => {
    if (typeof city !== "string") throw new Error("city must be a string");
    const weather = WEATHER[normalise(city)];
    return weather ? { city, ...weather } : { city, error: "brak danych dla tego miasta" };
  },
};

const { client, model } = makeClient();
const result = await runToolLoop({
  client,
  model,
  tools: [getWeather],
  messages: [{ role: "user", content: "Jaka jest teraz pogoda w Krakowie i w Gdańsku? Gdzie jest cieplej?" }],
  onToolCall: (name, args, output) =>
    console.log(`tool: ${name}(${JSON.stringify(args)}) -> ${JSON.stringify(output)}`),
});

console.log(`rounds: ${result.rounds}, tool calls: ${result.toolCalls.length}`);
console.log(`assistant: ${result.answer.trim()}`);
