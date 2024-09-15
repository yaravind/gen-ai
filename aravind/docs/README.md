## Introduction

1. [Chapter 1 - What Is Generative AI](What_Is_Generative_AI.md)
2. [Chapter 2 - Building Capable Assistants](Building_Capable_Assistants.md)

## Tokens

Tokens are the smallest unit of text that a language model can understand.
Use [this tool](https://platform.openai.com/tokenizer) to understand how some OpenAI models tokenize
text: https://platform.openai.com/tokenizer.

```one plus one is two``` is tokenized as ```['one', 'plus', 'one', 'is', 'two']```. **5 tokens.**

## Models

To run LLM's locally, we can use `Ollama` or `GPT4All`.

### Ollama

We use `ollama` to manage local models. It stores the downloaded models under `~/.ollama/models` which contains both
model blobs and manifests.

- **Model blobs** are large binary objects that store the actual parameters and data of a machine learning model,
  essential for making predictions or further training.
- **Manifests** provide metadata and information about a machine learning model, including its architecture,
  hyperparameters, and version information, facilitating model selection and integration into production systems.

Download and install the desktop application: https://ollama.com/download

> You can get the list of models supported by ollama here: https://ollama.com/library

### GPT4All

Download and install the desktop application: https://docs.gpt4all.io/index.html
You can use the app to download the models. It stores the downloaded models under `~/.cache/gpt4all/` which contains
both
model blobs and manifests.
> You can get the list of models supported by ollama here: https://docs.gpt4all.io/gpt4all_desktop/models.html

### RAG

For a current Retrieval-Augmented Generation (RAG) project, fine-tuning every parameter to ensure high-quality outputs.

- Beyond adjusting temperature or token limits, tweaking parameters like top-k (how many answer options to consider),
  top-p (how random the choices are), and frequency penalties (to avoid repeating answers).
- Combine that with smart retrieval—like choosing between keyword, semantic or hybrid search and deciding how many
  documents to pull—and you're not just pulling data; you're shaping it into meaningful insights.
- Building the right Large Language Model RAG system is a lot about experimentation—there’s no one-size-fits-all.

## Tools

All tools (DuckDuckGo Search, wikipedia etc.) have their specific purpose that is part of the description. **This
description is passed to the LLMS to provide context and help them generate better responses.**