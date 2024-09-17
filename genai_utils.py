import json


def pretty_print_json(result):
    print(json.dumps(result.model_dump(mode="json"), indent=2))
