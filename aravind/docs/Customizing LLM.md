# 1. Customizing LLMs

## 1.1. Introduction

The process of adapting a model for a certain task or making sure that the model outputs corresponds to what we expect
is called **conditioning**. Following are 2 techniques to condition a model:

1. Fine-tuning
2. Prompting

Conditioning can enhance capabilities in terms of the following and guide the model's behavior to be in line with what
is considered ethical and appropriate:

- task relevance
- specificity and
- coherence

**Alignment** is the process of ensuring that the model's general behavior, decision-making processes, and outputs
conform to broader human values, ethical principles, and safety considerations.

## 1.2. Conditioning Methods

| Stage                       | Technique                       | Examples                                      |
|-----------------------------|---------------------------------|-----------------------------------------------|
| Training                    | Data curation                   | Training on diverse data                      |
| Training                    | Objective function              | Careful design of training objective          |
| Training                    | Architecture & training process | Optimizing model structure & training         |
| Fine-tuning                 | Task specialization             | Training on specific tasks/datasets           |
| Inference-time conditioning | Dynamic inputs                  | Prefixes, control codes, and context examples |
| Human oversight             | Human-in-the-loop               | Human review and feedback                     |

## 1.3. Fine-tuning

In fine-tuning, the LM is trained on many examples of tasks formulated as natural language instructions, along with
appropriate responses. This is often done through reinforcement learning with human feedback.
Fine-tuning involves **training the pre-trained base model** on specific tasks or datasets relevant to the desired
application. This is done by updating the weights/parameters of the model using the new data and objectives. This
enables

- knowledge transfer from the general while customizing it for specialized tasks
- the model to adapt, becoming more accurate and contextually relevant to the intended use-case.

Fine-tuning can be resource-intensive, presenting the trade-off high performance and computational efficiency. These
limitations are addressed by strategies like:

- Adapters
- LoRA (Low-Rank Adaptation)

### 1.3.1. RLHF - Reinforcement Learning with Human Feedback

Human feedback serves as a critical signal to guide the model's learning process. By using RLHF, we can harness the
nuanced understanding of human evaluators to further refine the model behavior, ensuring outputs that are not only
relevant and accurate but also align with user intent and expectations. E.g. `InstructGPT`

RLHF has 3 main steps:

1. Supervised pre-training: The LM is first trained via standard supervised learning on human demonstrations.
2. Reward model training: A reward model is trained on human ratings of LM outputs to estimate a reward.
3. RL fine-tuning: The LM is fine-tuned using RL to maximize the expected reward from the reward model using an
   algorithm like PPO (**Proximal Policy Optimization**).

### 1.3.2. LoRA - Low-Rank Adaptation

LoRA is a technique that involves **updating only a subset of the model's parameters** to adapt to a new task or
dataset. This reduces the computational, memory, and storage costs while improving performance in **low-data and
out-of-domain** scenarios. Following methods help in achieving this:

- PEFT - Parameter Efficient Fine-Tuning: Enable the use of small checkpoints for each task, making the models more
  portable. This small set of trained weights can be added on top of the base LLM, allowing the same model to be used
  for multiple tasks without replacing the entire model.
- LoRA: It is a type of PEFT, where the pre-trained model weights are frozen. It introduces trainable rank decomposition
  matrices into each layer of the transformer architecture to reduce the number of trainable parameters.

### 1.3.3. Fine-tuning Limitations

- All models might **not be accessible** for fine-tuning
- **Insufficient data** either for the downstream task or domain
- **Dynamic data** for some domains like news etc.

## 1.4. Inference-time conditioning

Following are the techniques to condition the model during inference time. These techniques involve providing contextual
information during inference time, such as for in-context learning or retrieval augmentation:

- Prompt tuning
- Prefix tuning: Prepending trainable vectors to LLM layers.
- Constraining tokens: Forcing inclusion/exclusion of certain words.
- Metadata: Providing high-level info like genre, target audience, and so on.

### 1.4.1. Prompt tuning

Prompting is a technique that involves providing additional input or context **during inference time**. This is also
known as in-context learning. A prompt consists of three main parts:

1. **Instructions**: These are the guidelines or directives given to the model to guide its output. It generally
   describes
   the task requirements, goals, and format of input/output.
2. **Examples** (or demonstrations) provide diverse demonstrations of how different outputs should map to inputs. This
   allows the model to infer intentions and goals purely from demonstrations.
3. **Input** that the model should act on to generate the output.

#### Techniques

| Technique             | Key ideas                                                                                                                                                                                          |
|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Zero-shot             | - Leverages the models' pretraining<br/> - No examples provided                                                                                                                                    |
| Few-shot              | - Shows desired reasoning format                                                                                                                                                                   |
| Chain-of-Thought      | - Gives the model space to reason before answering<br/>- The performance was found to be proportional to the size of the model and improvements were negligible or even negative in smaller models |
| Least-to-Most         | - Prompts the model for simpler subtasks first<br/> - Decomposes a problem into smaller pieces                                                                                                     |
| Self-consistency      | - Picks the most frequent answer from multiple samples<br/>- Good use-cases is in the context of fact verification or information synthesis, where accuracy is paramount.                          |
| Chain-of-Density      | - Iteratively creates dense summaries by adding entities <br/> - Generates rich, concise summaries                                                                                                 | 
| Chain-of-Verification | - Verifies an initial response by generating and answering questions <br/> - Mimics human verification                                                                                             |
| Active prompting      | - Uses human feedback to guide the model's learning process <br/> - Finds effective few-shot examples                                                                                              |
| Tree-of-Thought       | - Generates and automatically evaluates multiple responses <br/> - Allows backtracking through reasoning paths                                                                                     |
| Verifiers             | - Trains a separate model to evaluate responses <br/> - Filters out incorrect responses                                                                                                            |
| Fine-tuning           | - Fine-tunes on an explanation dataset generated via prompting <br/> - Improves the model's reasoning capabilities                                                                                 |

**Notes:**

- To choose examples tailored to each input, `FewShotPromptTemplate` can accept `SemanticSimilarityExampleSelector`,
  based on embeddings rather than hardcoded examples.
- Self-consistency leverages the model's ability to reason and utilize internal knowledge while reducing the risk of
  outliers or incorrect information, _by
  focusing on the most recurring answer_.

### 1.4.2. Prefix tuning

In this, continuous task-specific vectors are trained and supplied to models at inference time.

- PELT (Parameter Efficient Transfer Learning): It involves training a small set of task-specific parameters that can
  be added to the base model at inference time.
- LST (Ladder Side-Tuning): It involves training a small set of task-specific parameters that can be added to the base
  model at inference time.