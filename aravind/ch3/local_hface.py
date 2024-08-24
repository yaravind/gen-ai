from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os
import torch

from langchain import PromptTemplate, LLMChain

# Try to cache the model

gen_text = pipeline(
    model="aisquared/dlite-v1-355m",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    framework="pt"
)

resp = gen_text("In this chapter, we'll discuss first steps with generative AI in Python.")
print(resp)

# Use HuggingFace's local transformers in LangChain
template = """Question: {question}
Answer: Let's think step by step.
"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=gen_text, verbose=True)

question = "What is electroencephalography?"
print(llm_chain.run(question))
