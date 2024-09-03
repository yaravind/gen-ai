## Building Capable Assistants

- To instill greater intelligence, productivity and trustworthiness in LLMs, we need to enhance LLMs through prompts,
  tools and structured reasoing techniques.

### Hallucination

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

## Summarization

For summarizing few sentences, basic prompting "Summarize this text: {text}" works but for longer texts, it is better
to use Chain of Density (CoD) method to incrementally increase the information density. This method involves breaking
the text into smaller parts and summarizing each part separately. The summaries are then combined to form the final
summary.

This repeated rewriting under length constraint forces increasing abstraction, fusion of details, and compression to
make room for additional entities in each step. This method is implemented in LangChain as `LLMSummarizerChain`.