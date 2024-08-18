from config import set_environment
from langchain.llms import FakeListLLM
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType


tools = load_tools(["python_repl"])
responses = ["Action: Python_REPL\nAction Input: print(2 + 3)", "Final Answer: 4"]
fake_llm = FakeListLLM(responses=responses)

agent = initialize_agent(tools, fake_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("whats 2 + 2")