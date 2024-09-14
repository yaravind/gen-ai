## 1. Building Capable Assistants

- To instill greater intelligence, productivity and trustworthiness in LLMs, we need to enhance LLMs through prompts,
  tools and structured reasoing techniques.

### 1.1. Hallucination

This can be addressed by automating fact-checking and source verification. By verifying claims against the available
evidence, we can reduce the spread of misinformation. Fact-checking involves three main steps:

1. Identifying the claim: Identify parts needing verification
2. Evidence retrieval: Find sources supporting or refuting the claim
3. Verdict prediction: Assess claim veracity based on evidence

LangChain provides `LLMCheckerChain` to automate fact-checking. It uses prompt chaining where the model is prompted
sequentially - first, to make the assumptions explicit. Next, the assumptions are fed back to the model in order to
check them one by one and then finally the model is tasked to make the final judgement.

Example:

1. Prompt: "Here is a statement: {statement}\nMake a bullet point list of the assumptions you made when producing the
   above statement."
2. Feed back assumptions to model: "Here is a bullet point list of assertions:\n {assertions}\n For each assertion,
   determine whether it is true or false. If it is false, explain why."
3. Final judgement: "In light of the aboce facts, how would you answer the question: {question}?"

> The above method doesn't guarantee correct answers, but it can put a stop to some incorrect results

## 2. Summarization

For summarizing few sentences, basic prompting "Summarize this text: {text}" works but for longer texts, it is better
to use Chain of Density (CoD) method to incrementally increase the information density. This method involves breaking
the text into smaller parts and summarizing each part separately. The summaries are then combined to form the final
summary.

This repeated rewriting under length constraint forces increasing abstraction, fusion of details, and compression to
make room for additional entities in each step. This method is implemented in LangChain as
`from langchain.chains.summarize import load_summarize_chain`.

### 2.1. Map-Reduce pipeline

To summarize long documents, we can split the document into smaller chunks that are suitable for the token context
length of the LLM and them apply map-reduce as follows:

1. Map: Each chunk is passed through a summarization chain
2. Collapse: Summarized chunks are combined into a single document
3. Reduce: The collapsed document is passed through the final LLM chain to produce output

Advantages:

- Allows parallel processing
- Enables the use of LLMs for reasoning, generating etc.

## 3. Monitor

For any serious usage of Gen AI, we need to understand the

- Capabilities
- pricing options
- use-case for different language models

| model                                                    | capabilities                                                                                                         |
|----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| GPT-3.5-Turbo                                            | specialize in dialogue ppa (chatbots, virtual assistants), capable of generating responses with accuracy and fluency |
| Instruct GPT (Ada: speed, Davinci: Complex instructions) | specialize in single-turn instruction following, capable of generating code, recipes, etc.                           |
| Codex                                                    | specialize in multi-turn instruction following, capable of generating code, recipes, etc.                            |

## Instruction Tuning & Function Calling

OpenAI's function calling builds on *instruction tuning*. By describing function in a schema, developers can tune LLMs
to return *structured outputs* adhering to that schema. Use-cases:

- Information Extraction: This capability can be used to *extract entities* from a text by outputting them in a
  predefined JSON format. We can extract specific entities and their properties using OpenAI *chat models*.
- Create chatbots that can answer questions using external tools or OpenAI plugins.
- Convert natural language queries int API calls or database queries.

Supported OpenAI models: `gpt-4-0613`, `gpt-3.5-turbo-0613`

Developers define the function to the model using JSON schema using the `/v1/chat/completions` endpoint.

> Instruction tuning and function calling allow models to produce callable code. This leads to tool integrations, where
> LLM agents can execute these function calls to connect LLMs with live data, services and runtime environments.