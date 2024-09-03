import openai
import os

from langchain.schema import LLMResult

from config import set_environment
from langchain.llms import OpenAI

set_environment()

api_key = os.getenv("OPENAI_API_KEY")

# List available models
openai.api_key = api_key
models = openai.Model.list()

# Print model IDs
for model in models['data']:
    print(model['id'])

openai_llm = OpenAI(temperature=0., model_name="gpt-4o-mini", openai_api_key=api_key)

resp: LLMResult = openai_llm.generate(prompts=["whats 2 + 2"])
print(resp)
