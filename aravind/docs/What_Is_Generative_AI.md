## What Is Generative AI

> GPT is **G**enerative **P**re-trained **T**ransformer.

Gen AI refers to class of algorithms that can _generate novel_ content (based on patterns learned from data), as opposed to analyzing or acting on _existing_ data like more traditional, predictive machine learning or AI systems. These Gen AI models also provide wide functionality including semantic search, content manipulation, and classification. The key difference between generative, over traditional models, is that generative models synthesize new data rather than just making predictions or decisions.

Gen AI models can handle different data modalities (text, image, video, music). Gen AI models _with images_ can create visual content, detect objects, segment images, caption etc.

| Domain                                                        | Examples                           |
|---------------------------------------------------------------|------------------------------------|
| Text-to-text                                                  | LlaMa, GPT 4, PaLM 2               |
| Text-to-image                                                 | DALL-E 2, Stable Diffusion, Imagen |
| Text-to-audio                                                 | Jukebox, AudioLM, MusicGen         |
| Text-to-video                                                 | Phenaki, Emu Video                 |
| Text-to-speech                                                | WaveNet, Tacotron                  |
| Speech-to-text (AKA Automatic Speech Recognition)             | Whisper, SpeechGPT                 |
| Image-to-text                                                 | DALL-E 3, CLIP                     |
| Image-to-image (Super resolution, style transfer, inpainting) | DALL-E 3, Stable Diffusion         |
| Text-to-code                                                  | Codex, Copilot, Code Interpreter   |
| Video-to-audio (Generate matching audio)                      | Soundify                           |

**Zero-shot** means the model is prompted with the question, while **5-shot** means, models were additionally given 5
question-answer examples.

**Post-training fine-tuning** of the models based on human interactions teaches the model how to perform a task by
providing demonstrations and feedback.

**Language Models** are statistical models used to predict the next words, or even a sentence based on the previous ones in a sequence of natural language. Some models use deep learning and are trained on massive datasets becoming **LLM**'s.

**Language Modeling** serves as a way of encoding the rules and structure of a language in a way that can be understood by machine. LLMs capture the structure of human language in terms of grammar, syntax, and semantics. Language modeling (and NLP) relies heavily on the quality of representational learning.

**Representational Learning** is about a model learning its internal representations of raw data to perform ML tasks, rather than relying only on engineered features.

Key challenges:

- data availability
- compute requirements
- bias in data
- evaluation difficulties
- potential misuse and
- other societal impacts

> The **backpropagation** algorithm introduced in the 1980s provided a way to effectively train multi-layer neural networks

**Importance of the number of parameters in an LLM**

The more parameters (or size) a model has, the higher its capacity (or potential) to capture the relationships between words and phrases as knowledge. Generally, the more parameters a model has the lower is its **perplexity** (how well a model can approximate the training dataset). _It seems that in models with between 2 and 7 billion parameters, new capabilities emerge such as ability to generate different creative test in formats like poems, code, emails, letters etc._

**What are tokens?**

The number of tokens reflects the breadth of knowledge an LLM has been exposed to during training. A higher token count generally indicates the LLM has been trained on a larger dataset, potentially giving it a wider vocabulary and understanding of various topics. In essence:

- Tokens represent the LLM's knowledge base (the what).
- Parameters represent the LLM's processing power (the how).

| Model  | Parameters                             | Tokens                     | 
|--------|----------------------------------------|----------------------------| 
| GPT 4  | 1.76 T                                 | 128K (input)/4096 (output) |
| Gemini | 175 T (Ultra), 50 T (Pro), 10 T (Nano) | 32K (input)/8192 (output)  |

## Technology evolution

- 1980: Backpropagation. Provided a way to effectively train multi-layer neural networks
- 2010: Autoencoders. A kind of neural network that can learn to compress data from the input layer to a representation, and then reconstruct the input. 
- 2013: Autoencoders served as the basis for VAEs (Variational Autoencoders). VAEs, unlike traditional Autoencoders, use variational inference to learn the distribution of data, also called the **latent space of input data**.
- 2014: GANs. 
- 2017: Transformer is a Deep Learning architecture that comprises self-attention and feed-forward neural networks, allowing it to effectively capture the word relationships in a sentence. Transformers build upon significant enhancements made in fundamental algorithms such as better optimization methods, more sophisticated model architectures, and improved regularization techniques. Transformers rely on **attention mechanisms** and resulted in further leap in performance of generative models.

**Transfer Learning or Representational Transfer** allows a model pre-trained on one task to be fine-tuned on another, similar task.