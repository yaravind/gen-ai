from config import set_environment
from langchain.llms import Ollama
import os

local_cache = "/Users/O60774/.ollama/models"

print("Downloaded models for Ollama:\n")
print(os.listdir(local_cache))

prompt = "In which country is Delhi?"

llm = Ollama(model="llama3.1", verbose=True)
resp = llm.invoke(prompt)
print("llama3.1: "+ resp)

llm = Ollama(model="llama2", verbose=True)
resp = llm.invoke(prompt)
print("llama2: "+ resp)

llm = Ollama(model="mistral", verbose=True)
resp = llm.invoke(prompt)
print("mistral: "+ resp)

llm = Ollama(model="phi3.5", verbose=True)
resp = llm.invoke(prompt)
print("phi3.5: "+ resp)

llm = Ollama(model="duckdb-nsql", verbose=True)
resp = llm.invoke("get passenger count, trip distance and fare amount from taxi table and order by all of them")
print("duckdb-nsql: "+ resp)

print(llm.invoke("Tell a joke"))