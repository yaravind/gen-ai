import openai
import os

from config import set_environment
from langchain.llms import OpenAI

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

set_environment()

api_key = os.getenv("OPENAI_API_KEY")

# List available models
openai.api_key = api_key
models = openai.Model.list()

# Print model IDs
for model in models['data']:
    print(model['id'])

tools = load_tools(["python_repl"])
responses = ["Action: Python_REPL\nAction Input: print(2 + 3)", "Final Answer: 4"]

openai_llm = OpenAI(temperature=0., model_name="gpt-4o-mini", openai_api_key=api_key)

agent = initialize_agent(tools, openai_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
resp = agent.run("whats 2 + 2")
print("--->" + resp)
