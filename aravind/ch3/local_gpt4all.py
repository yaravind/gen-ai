from langchain.llms import GPT4All
import os

local_cache = "/Users/O60774/Library/Application Support/nomic.ai/GPT4All/"

print("Downloaded models for GPT4All:\n")
print(os.listdir(local_cache))

llm = GPT4All(model=local_cache + "Meta-Llama-3-8B-Instruct.Q4_0.gguf", n_threads=8)
resp = llm("we can run LLM locally for all kinds of applications")
print(resp)
