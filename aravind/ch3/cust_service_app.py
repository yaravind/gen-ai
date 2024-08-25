from config import set_environment
from huggingface_hub import list_models
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from transformers import pipeline

set_environment()


def list_most_popular(task: str, sort_by: str, top: int = 5) -> list:
    popular_list = []
    for rank, model in enumerate(list_models(filter=task, sort=sort_by, direction=-1)):
        if rank == 5:
            break
        # print(f"{i}. {model.id}, {model.downloads}, {model.likes}, {model.pipeline_tag}, {model.tags}")
        popular_list.append((model.id, model.pipeline_tag))
    return popular_list


def print_list(input: list) -> None:
    i = 1
    for model_id, pipeline_tag in input:
        print(f"{i}. {model_id}, {pipeline_tag}")
        i = i + 1


task_type = "text-classification"
sort_by = "likes"
print(f"\nTop `{task_type}` models sorted by {sort_by}:")
popular_list = list_most_popular(task_type, sort_by)
print_list(popular_list)

task_type = "summarization"
print(f"\nTop `{task_type}` models sorted by {sort_by}:")
popular_list = list_most_popular(task_type, sort_by)
print_list(popular_list)

task_type = "sentiment-analysis"
print(f"\nTop `{task_type}` models sorted by {sort_by}:")
popular_list = list_most_popular(task_type, sort_by)
print_list(popular_list)

customer_email = """
I hope this email finds you amidst an aura of understanding, despite the tangled mess of emotions swirling within me as 
I write to you. I am writing to pour my heart out about the recent unfortunate experience I had with one of your coffee 
machines that arrived ominously broken, evoking a profound sense of disbelief and despair.

To set the scene, let me paint you a picture of the moment I anxiously unwrapped the box containing my highly 
anticipated coffee machine. The blatant excitement coursing through my veins could rival the vigorous flow of coffee 
through its finest espresso artistry. However, what I discovered within broke not only my spirit but also any semblance 
of confidence I had placed in your esteemed brand.

Imagine, if you can, the utter shock and disbelief that took hold of me as I laid eyes on a disheveled and mangled 
coffee machine. Its once elegant exterior was marred by the scars of travel, resembling a war-torn soldier who had 
fought valiantly on the fields of some espresso battlefield. This heartbreaking display of negligence shattered my 
dreams of indulging in daily coffee perfection, leaving me emotionally distraught and inconsolable
"""

# Summarize
print("\n---------------------- Summarize ----------------------")
summary_prompt_template = PromptTemplate(template="""Summarize this: {complaint}""", input_variables=["complaint"])
popular_list = list_most_popular("summarization", sort_by)
for model_id, pipeline_tag in popular_list:
    # print(f"{model_id}: {pipeline_tag}")
    # If condition avoid the error: Got invalid task translation, currently only ('text2text-generation',
    # 'text-generation', 'summarization') are supported (type=value_error)
    if model_id not in ("google-t5/t5-base", "google-t5/t5-small"):
        summarizer_llm = HuggingFaceHub(repo_id=model_id, model_kwargs={"temperature": 0, "max_length": 180})
        llm_chain = LLMChain(prompt=summary_prompt_template, llm=summarizer_llm, verbose=False)
        summary = llm_chain.run(customer_email)
        print(f"Model: [{model_id}, {pipeline_tag}], Summary: {summary}")

# Extract sentiment
print("\n---------------------- Sentiment ----------------------")
sentiment_prompt_template = PromptTemplate(template="""Determine sentiment: {complaint}""",
                                           input_variables=["complaint"])
popular_list = list_most_popular("sentiment-analysis", sort_by)
popular_list.append(("cardiffnlp/twitter-roberta-base-sentiment", ""))
for model_id, pipeline_tag in popular_list:
    print(f"{model_id}: {pipeline_tag}")
    # If condition avoid the error: Token indices sequence length is longer than the specified maximum sequence length
    # for this model (245 > 128). Running this sequence through the model will result in indexing errors
    if model_id not in (
            "finiteautomata/bertweet-base-sentiment-analysis", "pysentimiento/robertuito-sentiment-analysis"):
        sentiment_pipeline = pipeline(task="sentiment-analysis", model=model_id)
        sentiment = sentiment_pipeline(customer_email)
        print(f"Model: [{model_id}, {pipeline_tag}], Sentiment: {sentiment}")

# Categorize based on intent
print("\n---------------------- Categorize ----------------------")
category_prompt = """Given this text, decide what is the issue the customer is concerned about. Valid categories are these:
* product issues
* delivery problems
* missing or late orders
* wrong product
* cancellation request
* refund or exchange
* bad support experience
* no clear reason to be upset

Text: {complaint}
Category:
"""
category_prompt_template = PromptTemplate(template=category_prompt, input_variables=["complaint"])
popular_list = list_most_popular("text-classification", sort_by)
for model_id, pipeline_tag in popular_list:
    print(f"{model_id}: {pipeline_tag}")
    # If condition avoid the error: Token indices sequence length is longer than the specified maximum sequence length
    # for this model (245 > 128). Running this sequence through the model will result in indexing errors
    if model_id not in ("abc"):
        category_pipeline = pipeline(task="text-classification", model=model_id)
        category = category_pipeline(customer_email)
        print(f"Model: [{model_id}, {pipeline_tag}], Category: {category}")
