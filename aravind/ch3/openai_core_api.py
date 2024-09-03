import openai
import os

from config import set_environment

set_environment()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key
completion = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "write a haiku about ai"}
    ]
)
print(completion.choices[0].message['content'])