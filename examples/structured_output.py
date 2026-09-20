import logging
from enum import Enum

from pydantic import BaseModel
from termcolor import colored

from _vllm_env import make_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

client, model = make_client()
logging.info("Using model: %s", model)


class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"


class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType


def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
    }

    for message in messages:
        base_color = role_to_color.get(message["role"], "white")
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", base_color))
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", base_color))
        elif message["role"] == "assistant":
            print(colored(f"assistant: {message['content']}\n", base_color))
        else:
            print(colored(str(message), base_color))


def chat_completion_request(messages, extra_body=None):
    try:
        body = {
            "chat_template_kwargs": {"reasoning_effort": "none"},
        }
        if extra_body:
            body.update(extra_body)
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            extra_body=body,
        )
    except Exception as e:
        logging.warning("Unable to generate ChatCompletion response. Exception: %s", e)
        return e


def add_turn(prompt, messages, extra_body=None):
    messages.append({"role": "user", "content": prompt})
    chat_response = chat_completion_request(messages, extra_body)
    if isinstance(chat_response, Exception):
        logging.error("API error: %s", chat_response)
        messages.append({"role": "assistant", "content": f"API Error: {chat_response}"})
        return
    assistant_message = chat_response.choices[0].message
    messages.append(assistant_message.model_dump())


if __name__ == "__main__":
    json_schema = CarDescription.model_json_schema()
    logging.info("Configured JSON schema: %s", json_schema)

    messages = []
    add_turn("Wymyśl i napisz mi krótkie motywujące zdanie na dziś", messages)
    add_turn(
        "Wygeneruj JSON zawierający markę, model i typ nadwozia najbardziej ikonicznego samochodu z lat 90.",
        messages,
        {"guided_json": json_schema},
    )
    add_turn("Napisz teraz krótki motywujący tekst biorąc pod uwagę ten samochód", messages)
    add_turn(
        "Jaki jest najlepszy samochód dla 4 osobowej rodziny w Polsce? Odpowiedz w formacie JSON podając markę, model i typ nadwozia",
        messages,
        {"guided_json": json_schema},
    )

    logging.info("Messages:")
    pretty_print_conversation(messages)
