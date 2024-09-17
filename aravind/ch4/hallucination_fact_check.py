import json
import os

from langchain_core.messages import AIMessage

from genai_utils import pretty_print_json
from langchain_community.retrievers import WikipediaRetriever
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config import set_environment

set_environment()

# Load the input document
document_text = """The Eiffel Tower in Paris was completed in 1889.
The capital of Australia is Sydney.
Water boils at 100 degrees Celsius at standard atmospheric pressure.
Albert Einstein won the Nobel Prize in Physics in 1921 for his theory of relativity.
The Amazon Rainforest produces approximately 20% of the world's oxygen.
The Great Wall of China is visible from space.
The human body contains 206 bones.
Mount Everest is the tallest mountain in the world, with a peak at 29,029 feet above sea level.
The first human to set foot on the moon was Neil Armstrong in 1969.
The population of Tokyo, Japan, is over 30 million people."""
#document_text = """The Eiffel Tower in Paris was completed in 1889."""


def get_claims() -> list[str]:
    claims_list = document_text.split("\n")
    return claims_list


def get_evidence(claims: list[str]):
    retriever = WikipediaRetriever()
    evidence_list = []
    for claim in claims:
        evidences: list[Document] = retriever.invoke(claim)
        evidence = evidences[0].page_content if evidences else "No evidence found."
        # print(f"Evidence: {evidence}")
        evidence_list.append(evidence)
    return evidence_list


def evaluate_claim_evidence(in_claim, in_evidence):
    result = llm_chain.invoke({"claim": in_claim, "evidence": in_evidence})
    pretty_print_json(result)
    return result


# Verification logic
def verify_claim(in_claim, in_evidence):
    evaluation_result: AIMessage = evaluate_claim_evidence(in_claim, in_evidence)
    result_json = evaluation_result.model_dump(mode="dict")
    # print(f"Evaluation Result: {result_json}")
    # Analyze the evaluation result to determine if the claim is true, false, or uncertain
    return extract_evaluation_details(result_json)


def extract_evaluation_details(response_json):
    # Parse the content field which is a JSON string
    content = json.loads(response_json["content"])
    return content.get("accuracy"), content.get("explanation"), content.get("confidenceLevel")


# Extract claims from the input document
claims = get_claims()
print(f"Claims count: {len(claims)}")

# Retrieve evidence for claims
evidences = get_evidence(claims)
print(f"Evidence count: {len(evidences)}")

# Fact check
prompt = """You are an expert fact-checker. Your task is to evaluate the following claim against the provided context 
and determine its accuracy. Provide a thorough explanation for your decision.

### Claim:
{claim}

### Context:
{evidence}

### Evaluation Criteria:
1. accuracy: Determine if the claim is true, false, or partially true based on the context.
2. explanation: Provide a detailed explanation of your reasoning, referencing specific parts of the context.
3. confidenceLevel: Rate your confidence in the evaluation on a scale from 1 to 10, with 10 being completely confident.

Please respond in JSON format with the following keys:
    "accuracy": [True/False/Partially-True/Unverifiable]
    "explanation": [Your detailed explanation]
    "confidenceLevel": [1-10].
"""
prompt_template = PromptTemplate(template=prompt, input_variables=["claim", "evidence"])
llm = ChatOpenAI(temperature=0., model_name="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"), verbose=True)
llm_chain = prompt_template | llm

for claim, evidence in zip(claims, evidences):
    accuracy, explanation, confidence_level = verify_claim(claim, evidence)
    print('-' * 80)
    print(f"Claim: {claim}")
    print(f"Evidence: {evidence}")
    print(f"Accuracy: {accuracy}")
    print(f"Explanation: {explanation}")
    print(f"Confidence Level: {confidence_level}")
