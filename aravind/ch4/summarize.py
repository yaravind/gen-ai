from config import set_environment
#from langchain_decorators import llm_prompt
from langchain import PromptTemplate, OpenAI
from langchain.schema import StrOutputParser


def summarize2(text: str, length="short") -> str:
    prompt = PromptTemplate.from_template("""
    Summarize this text in {length} length:

    {text}
    """)
    llm = OpenAI(model="davinci-002")
    runnable = prompt | llm | StrOutputParser()
    summary = runnable.invoke({"text": text, "length": length})
    return summary


# @llm_prompt
# def summarize(text: str, length="short") -> str:
#     """
#     Summarize this text in {length} length:
#
#     {text}
#     """
#     return


set_environment()

text = """We use `ollama` to manage local models. It stores the downloaded models under `~/.ollama/models` which contains both
model blobs and manifests.

- **Model blobs** are large binary objects that store the actual parameters and data of a machine learning model,
  essential for making predictions or further training.
- **Manifests** provide metadata and information about a machine learning model, including its architecture,
  hyperparameters, and version information, facilitating model selection and integration into production systems."""

summary = summarize2(text)
print(summary)

# summary = summarize(text)
# print(summary)
