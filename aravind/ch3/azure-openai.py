import os

from openai import AzureOpenAI
from config import set_environment

set_environment()

openai_client = AzureOpenAI(
    azure_endpoint="https://ai.ko.com/",
    azure_deployment="gpt-4o",
    api_version="2024-02-01",
    api_key=os.getenv("TCCC_AI_API_PRIMARY_KEY"),
    default_headers={"OCP-Apim-Subscription-Key": os.getenv("TCCC_AI_API_PRIMARY_KEY")}
)
messages = [{"role": "user", "content": "write a 2 paragraph poem on coca-cola"}]
completion = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0.7,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stream=False,
    max_tokens=1024
)
print(completion.choices[0].message.content)
