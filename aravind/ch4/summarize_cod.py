"""summarization prompts.

This is inspired by https://github.com/daveshap/Quickly_Extract_Science_Papers
"""
import logging
import os
import time
from pathlib import Path

from langchain import PromptTemplate, LLMChain, HuggingFaceHub
from langchain.callbacks import get_openai_callback
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document

from config import set_environment
from langchain_utils import rate_limiter
from langchain.chat_models import ChatOpenAI

# Initialize logging
logging.basicConfig(encoding="utf-8", level=logging.INFO)
LOGGING = logging.getLogger()

# Create Chain of Density Prompts
DENSITY_PROMPT = """
Article: {text}

---
Article Summary Guidelines
You will generate increasingly concise, entity-dense summaries of the above article. Repeat the following 2 steps 5 times.

Repeat the following 2 steps 5 times:
Step 1: Identify 1-3 informative entities (";" delimited) from the previous generated version of the summary. 
Step 2: Write a new, denser summary of shorter length that covers every entity mentioned, plus missing entities.

A missing entity is:
- Relevant: to the main story.
- Specific: descriptive yet concise (5 words or fewer).
- Novel: not in the previous summary.
- Faithful: present in the article.
- Anywhere: located anywhere in the article.

Guidelines:
- Make every word count!
- Make space with fusion, compression, and removal of uninformative phrases
- The summaries should become highly dense and concise yet self-contained.
- Missing entities can appear anywhere in the new summary. 
- Never drop entities from the previous summary. If space cannot be made, add fewer new entities.

Answer in JSON. The JSON should be a list (length 5) of dictionaries whose keys are "Missing_Entities" and "Denser_Summary".
"""

SUMMARY = (
    "Summarize this text in as much detail as possible. Give a clear explanation of the objectives, core assertions, implications, "
    "and mechanics elucidated in this text - remove citations! \n"
    "I want to highlight the main points starting from their importance and impact, mechanics, available tools, and potential extensions that impact dimensions such as privacy, safety flexibility, competitive performance, ease of use."
    "Text: {text} \n"
)

HIGH_LEVEL = (
    "Please explain the value of this text in basic terms like you're "
    "talking to a CEO. So what? What's the bottom line here?\n"
    "{text}\n"
)

ANALOGY = (
    "Please give me an analogy or metaphor that will help explain this text "
    "to a broad audience!\n"
    "{text}\n"
)


def load_and_split_pdf(pdf_file_path: str) -> list[Document]:
    """Read and split pdf document."""
    pdf_loader = PyPDFLoader(pdf_file_path)
    return pdf_loader.load_and_split()


def summarize_pdf(docs: list[Document]) -> str:
    """Summarize a list of Documents.

       Result is returned as dict with these keys:
       * 'intermediate_steps' (list[str]),
       * 'output_text' (str), and
       * 'analogy' (str)
    """
    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            chain = load_summarize_chain(
                HUGGING,
                chain_type="map_reduce",
                map_prompt=PromptTemplate(input_variables=["text"], template=SUMMARY),
                combine_prompt=PromptTemplate(input_variables=["text"], template=HIGH_LEVEL),
                return_map_steps=True
            )
            summary = chain({"input_documents": docs})
            # only works if the model is OpenAI
            # with get_openai_callback() as cb:
            #     llm_chain = LLMChain.from_string(
            #         llm=CHAT,
            #         template=ANALOGY
            #     )
            #     summary["analogy"] = llm_chain.predict(text=summary["output_text"])
            #     LOGGING.info(f"Total Tokens: {cb.total_tokens}")
            #     LOGGING.info(f"Prompt Tokens: {cb.prompt_tokens}")
            #     LOGGING.info(f"Completion Tokens: {cb.completion_tokens}")
            #     LOGGING.info(f"Total Cost (USD): ${cb.total_cost}")

            # For hugging face
            summary["analogy"] = HUGGING.predict(text=summary["output_text"])

            return summary
        except Exception as e:
            LOGGING.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retry_attempts - 1:
                LOGGING.info("Retrying after 25 seconds...")
                time.sleep(25)
            else:
                raise


def format_summary(summary: dict) -> str:
    """Format a summary into a single string."""
    summary_template = PromptTemplate(
        input_variables=["main_summary", "executive_summary", "analogy"],
        template="{main_summary}\nSUMMARY:\n{executive_summary}\nANALOGY: {analogy}"
    )
    return summary_template.format(
        main_summary="\n".join(summary["intermediate_steps"]),
        executive_summary=summary["output_text"],
        analogy=summary["analogy"]
    )


def create_pdf_summary(pdf_file):
    """Given a directory, create a summary as txt file.

    If summary was already created, return previous output.
    """
    pdf_path = Path(pdf_file)
    output_file = pdf_path.with_suffix('.txt')
    if os.path.exists(output_file):
        LOGGING.info("Summary file already exists!")
        with open(output_file, "r") as f:
            return f.read()

    summary = summarize_pdf(
        pdf_file_path=pdf_file
    )
    # write output:
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary)


# Set up the OpenAI key
set_environment()

# Create the model
CHAT = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, max_retries=2)
print("==> Using model: " + CHAT.model_name)

HUGGING = HuggingFaceHub(repo_id='tiiuae/falcon-7b-instruct',
                         huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
HUGGING.client.api_url = 'https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct'
pdf_file = "/Users/O60774/Downloads/Brief History of NLP.pdf"
docs = load_and_split_pdf(pdf_file)
# Iterate through the list of Document objects and print their metadata
for doc in docs:
    print(doc.metadata)

summary = summarize_pdf(docs)
formatted_summary = format_summary(summary)
# write output:
pdf_path = Path(pdf_file)
output_file = pdf_path.with_suffix('.txt')
with open(output_file, "w", encoding="utf-8") as f:
    f.write(formatted_summary)

print(summary["analogy"])
print(summary["intermediate_steps"])
print(summary["output_text"])
