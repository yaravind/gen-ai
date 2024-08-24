## Notes

1. [Chapter 1 - What Is Generative AI](What_Is_Generative_AI.md)

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