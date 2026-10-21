import logging

from termcolor import colored

from _vllm_env import make_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

client, model = make_client()
logging.info("Using model: %s", model)

role_to_color = {
    "system": "red",
    "user": "green",
    "assistant": "blue",
    "reasoning": "yellow",
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
            if message.get("reasoning") is not None:
                parts_to_print.append(
                    colored(
                        "[ Thinking... ]\n"
                        + message["reasoning"]
                        + "\n[ Thinking finished ]",
                        role_to_color.get("reasoning", "white"),
                    )
                )

            if message.get("content") is not None:
                parts_to_print.append(colored(message["content"], base_color))

            if not parts_to_print:
                print(colored("[No output from assistant]", base_color))
            else:
                print("".join(parts_to_print))
        else:
            print(colored(str(message), base_color))

        print()


def _delta_reasoning(delta) -> str | None:
    # vLLM may expose either field depending on version / parser.
    for attr in ("reasoning", "reasoning_content"):
        val = getattr(delta, attr, None)
        if val:
            return val
    return None


def chat_completion_request(messages, reasoning_effort: str = "none"):
    """Stream a chat completion.

    Uses v3.1 ``reasoning_effort`` (none|low|medium|high|max) via
    ``chat_template_kwargs``. Requires the Bielik v3.1 reasoning parser
    plugin when effort != none.
    """
    try:
        kwargs = {
            "model": model,
            "messages": messages,
            "stream": True,
            "extra_body": {
                "chat_template_kwargs": {"reasoning_effort": reasoning_effort},
            },
        }
        if reasoning_effort != "none":
            kwargs["temperature"] = 1.0
        else:
            kwargs["max_tokens"] = 2000
            kwargs["temperature"] = 0.2
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        logging.warning("Unable to generate ChatCompletion response. Exception: %s", e)
        return e


def process_streamed_response(stream, print_stream=False):
    full_response_content = ""
    full_reasoning_content = ""
    reasoning = False
    assistant_color = role_to_color.get("assistant", "white")
    reasoning_color = role_to_color.get("reasoning", "white")

    for chunk in stream:
        delta = chunk.choices[0].delta
        finish_reason = chunk.choices[0].finish_reason

        reasoning_piece = _delta_reasoning(delta)
        if reasoning_piece:
            full_reasoning_content += reasoning_piece
            if print_stream:
                if not reasoning:
                    print(colored("[ Thinking... ]\n", reasoning_color), end="", flush=True)
                print(colored(reasoning_piece, reasoning_color), end="", flush=True)
            reasoning = True
        elif getattr(delta, "content", None):
            full_response_content += delta.content
            if print_stream:
                if reasoning:
                    print(colored("[ Thinking finished ]\n", reasoning_color), end="", flush=True)
                    reasoning = False
                print(colored(delta.content, assistant_color), end="", flush=True)

        if finish_reason:
            break

    if full_reasoning_content and not full_response_content:
        # workaround: some stacks put the whole answer in reasoning only
        full_response_content = full_reasoning_content
        full_reasoning_content = ""
        if print_stream:
            if reasoning:
                print(colored("\n[ Thinking finished ]\n", reasoning_color), end="", flush=True)
                reasoning = False
            print(colored(full_response_content, assistant_color), end="", flush=True)

    if print_stream:
        print()

    assistant_message_dict = {"role": "assistant", "content": None, "reasoning": None}
    if full_reasoning_content:
        assistant_message_dict["reasoning"] = full_reasoning_content
    assistant_message_dict["content"] = full_response_content if full_response_content else ""
    return assistant_message_dict


def add_turn(prompt, messages, reasoning_effort: str = "none"):
    messages.append({"role": "user", "content": prompt})

    stream = chat_completion_request(messages, reasoning_effort)
    if isinstance(stream, Exception):
        logging.error("Error in API call: %s", stream)
        messages.append(
            {"role": "assistant", "content": f"API Error: Could not get response. {stream}"}
        )
        print(colored(f"assistant: API Error: Could not get response. {stream}", "red"))
        return

    print(colored("assistant: ", role_to_color.get("assistant")), end="", flush=True)
    assistant_response_dict = process_streamed_response(stream, print_stream=True)
    messages.append(assistant_response_dict)


if __name__ == "__main__":
    messages = []

    prompts = [
        "Wymyśl i napisz mi krótkie motywujące zdanie na dziś",
        "Jade na jednodniową wycieczkę do Końskich. Co warto zobaczyć?",
        "To teraz krótki motywujacy tekst na ten temat",
    ]
    # Turn 2 uses medium reasoning; others force-closed empty think (effort=none).
    efforts = ["none", "medium", "none"]

    for i, prompt in enumerate(prompts):
        effort = efforts[i]
        print(colored(f"\n--- Turn {i+1} (reasoning_effort={effort}) ---", "yellow", attrs=["bold"]))
        print(colored(f"user: {prompt}", role_to_color.get("user")))
        add_turn(prompt, messages, reasoning_effort=effort)

    logging.info("--- Final Conversation History ---")
    pretty_print_conversation(messages)
