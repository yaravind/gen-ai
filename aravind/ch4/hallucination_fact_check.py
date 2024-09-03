from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import LLMCheckerChain
from langchain.llms import Ollama
from config import set_environment

set_environment()

# Load the Ollama language model
llm = Ollama(model="mistral", verbose=True)
question = "What type of mammal lays biggest eggs?"

callback = StdOutCallbackHandler()

checker_chain = LLMCheckerChain.from_llm(llm, verbose=True)
checker_chain.callbacks = [callback]
resp = checker_chain.run(question)
print(resp)
