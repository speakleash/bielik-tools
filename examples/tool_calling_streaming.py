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
        role = message["role"]
        base_color = role_to_color.get(role, "white")
        print(colored(f"{role}: ", base_color), end="")

        if role == "system":
            print(colored(message.get("content", "[No content]"), base_color))
        elif role == "user":
            print(colored(message.get("content", "[No content]"), base_color))
        elif role == "assistant":
            parts_to_print = []
            if message.get("content") is not None:
                parts_to_print.append(colored(message["content"], base_color))

            if message.get("tool_calls"):
                if message.get("content"):
                    parts_to_print.append(colored("\n---\n", base_color))

                tool_calls_str_parts = []
                for tc_idx, tool_call in enumerate(message["tool_calls"]):
                    func_name = tool_call.get("function", {}).get("name", "[unknown function]")
                    func_args = tool_call.get("function", {}).get("arguments", "{}")
                    tool_id = tool_call.get("id", "[no id]")
                    tool_calls_str_parts.append(
                        colored(
                            f"Tool Call {tc_idx+1}: {func_name}(args={func_args}) ID: {tool_id}",
                            base_color,
                        )
                    )
                parts_to_print.append(
                    "\n  ".join(tool_calls_str_parts)
                    if len(tool_calls_str_parts) > 1
                    else "".join(tool_calls_str_parts)
                )

            if not parts_to_print:
                print(colored("[No output from assistant]", base_color))
            else:
                print("".join(parts_to_print))

        elif role == "tool":
            tool_name = message.get("name", "[unknown tool]")
            tool_id = message.get("tool_call_id", "[no id]")
            tool_content = message.get("content", "[No content from tool]")
            print(colored(f"(name: {tool_name}, tool_call_id: {tool_id}): {tool_content}", base_color))
        else:
            print(colored(str(message), base_color))

        print()


def chat_completion_request(messages):
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=500,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
            stream=True,
            extra_body={
                "chat_template_kwargs": {"reasoning_effort": "none"},
            },
        )
    except Exception as e:
        logging.warning("Unable to generate ChatCompletion response. Exception: %s", e)
        return e


def process_streamed_response(stream, print_stream=False):
    full_response_content = ""
    tool_call_deltas_aggregator = {}
    tool_call_in_progress_printed = False

    for chunk in stream:
        delta = chunk.choices[0].delta
        finish_reason = chunk.choices[0].finish_reason

        if delta.content:
            full_response_content += delta.content
            if print_stream:
                print(colored(delta.content, "blue"), end="", flush=True)

        if delta.tool_calls:
            if print_stream and not tool_call_in_progress_printed and not delta.content:
                print(colored("[tool call(s) in progress...]", "yellow"), end="", flush=True)
                tool_call_in_progress_printed = True

            for tool_call_chunk in delta.tool_calls:
                index = tool_call_chunk.index
                if index not in tool_call_deltas_aggregator:
                    tool_call_deltas_aggregator[index] = {
                        "id": None,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }

                if tool_call_chunk.id:
                    tool_call_deltas_aggregator[index]["id"] = tool_call_chunk.id

                if tool_call_chunk.type:
                    tool_call_deltas_aggregator[index]["type"] = tool_call_chunk.type

                if tool_call_chunk.function:
                    if tool_call_chunk.function.name:
                        tool_call_deltas_aggregator[index]["function"]["name"] += (
                            tool_call_chunk.function.name
                        )
                    if tool_call_chunk.function.arguments:
                        tool_call_deltas_aggregator[index]["function"]["arguments"] += (
                            tool_call_chunk.function.arguments
                        )

        if finish_reason:
            break

    if print_stream:
        print()

    assistant_message_dict = {"role": "assistant", "content": None}
    processed_tool_calls = []
    if tool_call_deltas_aggregator:
        for i in sorted(tool_call_deltas_aggregator.keys()):
            tc = tool_call_deltas_aggregator[i]
            if tc.get("id") and tc["function"].get("name"):
                processed_tool_calls.append(tc)
            else:
                logging.warning("Incomplete tool call data aggregated at index %s: %s", i, tc)

    if processed_tool_calls:
        assistant_message_dict["tool_calls"] = processed_tool_calls

    if full_response_content:
        assistant_message_dict["content"] = full_response_content
    elif not processed_tool_calls:
        assistant_message_dict["content"] = ""

    return assistant_message_dict


def call_function(name, args):
    logging.info("Attempting to call function: %s with args: %s", name, args)
    if name == "get_current_weather":
        location = args.get("location", "unknown location")
        logging.info("Simulating get_current_weather for %s", location)
        return json.dumps({"temperature": "25°C", "weather": "sunny", "location": location})
    if name == "get_n_day_weather_forecast":
        location = args.get("location", "unknown location")
        num_days = args.get("num_days", 1)
        logging.info("Simulating get_n_day_weather_forecast for %s, %s days", location, num_days)
        forecast_data = [
            {
                "day": i + 1,
                "temperature": f"{20 + i}°C",
                "weather": ("rainy", "cloudy", "windy and sunny")[i % 3],
            }
            for i in range(num_days)
        ]
        return json.dumps({"forecast": forecast_data, "location": location, "num_days": num_days})

    logging.warning("Function %s not found.", name)
    return None


def add_turn(prompt, messages):
    messages.append({"role": "user", "content": prompt})

    stream1 = chat_completion_request(messages)
    if isinstance(stream1, Exception):
        logging.error("Error in first API call: %s", stream1)
        messages.append(
            {"role": "assistant", "content": f"API Error: Could not get response. {stream1}"}
        )
        print(colored(f"assistant: API Error: Could not get response. {stream1}", "red"))
        return

    print(colored("assistant: ", role_to_color.get("assistant")), end="", flush=True)
    assistant_response_dict = process_streamed_response(stream1, print_stream=True)
    messages.append(assistant_response_dict)

    if assistant_response_dict.get("tool_calls"):
        tool_calls = assistant_response_dict["tool_calls"]
        function_response_messages_to_append = []

        for tool_call in tool_calls:
            tool_function_name = tool_call["function"]["name"]
            tool_function_args_str = tool_call["function"]["arguments"]
            tool_call_id = tool_call["id"]

            logging.info(
                "Received tool call: %s(args_str='%s') ID: %s",
                tool_function_name,
                tool_function_args_str,
                tool_call_id,
            )

            try:
                tool_function_args = json.loads(tool_function_args_str)
            except json.JSONDecodeError as e:
                logging.error(
                    "Invalid JSON arguments for %s: %s. Error: %s",
                    tool_function_name,
                    tool_function_args_str,
                    e,
                )
                result_content = json.dumps({"error": "Invalid JSON arguments", "details": str(e)})
            else:
                result_content = call_function(tool_function_name, tool_function_args)
                if result_content is None:
                    logging.warning(
                        "Function %s did not return a valid result (returned None).",
                        tool_function_name,
                    )
                    result_content = json.dumps(
                        {
                            "error": f"Function {tool_function_name} not found or did not return data."
                        }
                    )

            tool_response_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_function_name,
                "content": result_content,
            }
            function_response_messages_to_append.append(tool_response_message)

        for msg in function_response_messages_to_append:
            messages.append(msg)
            print(
                colored(
                    f"tool (name: {msg['name']}, tool_call_id: {msg['tool_call_id']}): {msg['content']}",
                    "magenta",
                )
            )

        stream2 = chat_completion_request(messages)
        if isinstance(stream2, Exception):
            logging.error("Error in second API call (after function results): %s", stream2)
            messages.append(
                {
                    "role": "assistant",
                    "content": f"API Error: Could not get response after tool call. {stream2}",
                }
            )
            print(
                colored(
                    f"assistant: API Error: Could not get response after tool call. {stream2}",
                    "red",
                )
            )
            return

        print(colored("assistant: ", role_to_color.get("assistant")), end="", flush=True)
        final_assistant_response_dict = process_streamed_response(stream2, print_stream=True)
        messages.append(final_assistant_response_dict)


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
