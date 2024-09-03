import os

from config import set_environment
from langchain.llms import VertexAI
from langchain import PromptTemplate, LLMChain
import langchain

from langchain.chat_models import ChatVertexAI
from langchain.prompts import ChatPromptTemplate

set_environment()

# Vertex AI
#print(f"Vertex AI SDK version: {aiplatform.__version__}")

system = "You are a helpful assistant that translates {input_language} to {output_language}."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
chain = prompt
resp = chain.invoke(
    {
        "input_language": "English",
        "output_language": "Chinese",
        "text": "It's Monday today",
    }
)

# To use model
model = VertexAI(model_name="gemini-pro")
message = "What are some of the pros and cons of Python as a programming language?"
model.invoke(message)

#print(resp)


# template = """
# Question: {question}
# Answer: Let's think step by step.
# """
# prompt = PromptTemplate(template=template, input_variables=["question"])
# #
# llm = VertexAI(project="gen-ai", credentials=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
# print(llm.client)
#
# llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
#
# question = "What NFL team won the super bowl in the year Justin Beiber was born?"
#
# resp = llm_chain.run(question)
# print(resp)
