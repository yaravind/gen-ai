from config import set_environment
from langchain.llms import OpenAI

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

set_environment()

tools = load_tools(["python_repl"])
responses = ["Action: Python_REPL\nAction Input: print(2 + 3)", "Final Answer: 4"]
openai_llm = OpenAI(temperature=0., model_name="gpt-4o-mini")

agent = initialize_agent(tools, openai_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("whats 2 + 2")

