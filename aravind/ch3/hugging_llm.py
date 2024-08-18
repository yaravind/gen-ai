from config import set_environment
from langchain.llms import HuggingFaceHub
import os

"""
Huggingface API endpoint docs: https://huggingface.co/docs/api-inference/index
"""

set_environment()

# The following uses default model and doesn't product the correct answer
prompt = "In which country is Delhi?"
# hugging_llm = HuggingFaceHub(model_kwargs={"temperature": 0.5, "max_length": 64}, repo_id="tiiuae/falcon-7b-instruct")
# hugging_llm.client.api_url = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
# print(hugging_llm.client)
#
# completion = hugging_llm(prompt)
# print(completion)

# The following produces the correct result but uses a different API
llm = HuggingFaceHub(repo_id='tiiuae/falcon-7b-instruct', huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
llm.client.api_url = 'https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct'
resp = llm.invoke(prompt)
print(resp)
