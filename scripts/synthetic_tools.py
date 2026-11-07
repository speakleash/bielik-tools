"""Deterministic, allowlisted test tools; no shell, files, network or eval."""
from collections import Counter


class SyntheticTools:
    def __init__(self, fail_first_weather=False):
        self.calls = []
        self.fail_first_weather = fail_first_weather

    def execute(self, name, arguments):
        if not isinstance(arguments, dict):
            raise ValueError("Arguments must be an object")
        if name == "add":
            if set(arguments) != {"a", "b"} or any(type(v) is not int for v in arguments.values()):
                raise ValueError("add needs exactly integer a and b")
            result = {"value": arguments["a"] + arguments["b"]}
        elif name == "weather":
            if set(arguments) != {"city"} or not isinstance(arguments["city"], str) or not arguments["city"]:
                raise ValueError("weather needs exactly a non-empty city string")
            previous = sum(call["name"] == name for call in self.calls)
            result = ({"error": "synthetic transient failure", "retryable": True}
                      if self.fail_first_weather and previous == 0 else
                      {"city": arguments["city"], "temperature_c": 21, "source": "synthetic"})
        else:
            raise ValueError(f"Unknown test tool: {name!r}")
        self.calls.append({"name": name, "arguments": dict(arguments), "result": result})
        return result

    def counts(self):
        return dict(Counter(c["name"] for c in self.calls))
