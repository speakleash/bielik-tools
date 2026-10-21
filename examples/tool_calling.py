import json
import logging

from termcolor import colored

from _vllm_env import make_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

client, model = make_client()
logging.info("Using model: %s", model)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                    },
                },
                "required": ["location", "num_days"],
            },
        },
    },
]
logging.info("Available tools: %s", len(tools))

role_to_color = {
    "system": "red",
    "user": "green",
    "assistant": "blue",
    "tool": "magenta",
}


def pretty_print_conversation(messages):
    for message in messages:
        role = message["role"] if isinstance(message, dict) else getattr(message, "role", "?")
        base_color = role_to_color.get(role, "white")
        if role == "system":
            print(colored(f"system: {message['content']}\n", base_color))
        elif role == "user":
            print(colored(f"user: {message['content']}\n", base_color))
        elif role == "assistant" and message.get("tool_calls"):
            print(colored(f"assistant: {message['tool_calls']}\n", base_color))
        elif role == "assistant" and not message.get("tool_calls"):
            print(colored(f"assistant: {message['content']}\n", base_color))
        elif role == "tool":
            print(colored(f"tool ({message['name']}): {message['content']}\n", base_color))
        else:
            print(colored(str(message), base_color))


def chat_completion_request(messages, tool_choice="auto"):
    """Chat completion with tools (needs Bielik v3.1 tool parser plugin)."""
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=500,
            tools=tools,
            tool_choice=tool_choice,
            temperature=0.2,
            extra_body={
                # Keep think force-closed so tool tags land in content for the parser.
                "chat_template_kwargs": {"reasoning_effort": "none"},
            },
        )
    except Exception as e:
        logging.warning("Unable to generate ChatCompletion response. Exception: %s", e)
        return e


def call_function(name, args):
    if name == "get_current_weather":
        return json.dumps({"temperature": "25°C", "weather": "sunny"})
    if name == "get_n_day_weather_forecast":
        return json.dumps(
            {
                "forecast": [
                    {"temperature": "21°C", "weather": "rainy"},
                    {"temperature": "22°C", "weather": "cloudy"},
                    {"temperature": "23°C", "weather": "windy and sunny"},
                ]
            }
        )
    return None


def add_turn(prompt, messages):
    messages.append({"role": "user", "content": prompt})
    chat_response = chat_completion_request(messages)
    if isinstance(chat_response, Exception):
        logging.error("API error: %s", chat_response)
        messages.append({"role": "assistant", "content": f"API Error: {chat_response}"})
        return

    assistant_message = chat_response.choices[0].message
    messages.append(assistant_message.model_dump())

    tool_calls = assistant_message.tool_calls
    if tool_calls:
        for tc in tool_calls:
            tool_call_id = tc.id
            tool_function_name = tc.function.name
            tool_function_args = json.loads(tc.function.arguments)
            logging.info("Function call %s(args=%s)", tool_function_name, tool_function_args)

            result = call_function(tool_function_name, tool_function_args)
            if not result:
                logging.warning("Function %s does not exist", tool_function_name)
                result = "{}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_function_name,
                    "content": result,
                }
            )
        chat_response = chat_completion_request(messages)
        if isinstance(chat_response, Exception):
            logging.error("API error after tools: %s", chat_response)
            messages.append({"role": "assistant", "content": f"API Error: {chat_response}"})
            return
        assistant_message = chat_response.choices[0].message
        messages.append(assistant_message.model_dump())


if __name__ == "__main__":
    messages = []

    prompts = [
        "Wymyśl i napisz mi krótkie motywujące zdanie na dziś",
        "A tak w ogóle to jaka dziś pogoda na dworze w Końskich?",
        "To teraz krótki motywujacy tekst biorąc pod uwagę pogodę",
        "A jaka będzie pogoda przez najbliższe 3 dni w Kielcach? Prognozę podaj w tabelce.",
        "Czy jutro w Kielcach przyda mi się parasol?",
        "A za trzy dni?",
    ]

    for i, p in enumerate(prompts):
        print(colored(f"\n--- Turn {i+1} ---", "yellow", attrs=["bold"]))
        print(colored(f"user: {p}", role_to_color.get("user")))
        add_turn(p, messages)

    logging.info("--- Final Conversation History ---")
    pretty_print_conversation(messages)
