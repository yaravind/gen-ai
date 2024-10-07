import json
import logging
import time

import nltk
import os

LOG = logging.getLogger('genai_utils')
from langchain_core.documents import Document


def pretty_json(input_dict: dict):
    return json.dumps(input_dict, indent=2)


def pretty_print_json(result):
    print(json.dumps(result.model_dump(mode="json"), indent=2))


def log_pretty_json(result):
    print(json.dumps(result.model_dump(mode="json"), indent=2))


def download_nltk_resource(path, resource):
    try:
        if not os.path.exists(os.path.join(nltk.data.find(path), resource)):
            nltk.download(resource)
    except LookupError:
        nltk.download(resource)


def log_elapsed_time(start_time: float, step_name: str = ""):
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    LOG.debug(f"Elapsed time for '{step_name}': ({int(minutes)} mins {seconds:.2f} secs)")
