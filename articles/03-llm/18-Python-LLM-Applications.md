# Day 18 - Model Families and LLM Applications

## Overview
Explore the diversity of Large Language Model (LLM) families, their architectures, strengths, and real-world Python integrations. Learn how to compare models, wrap APIs, and fine-tune LLMs.

## 🌟 Learning Objectives
By the end of this lesson, you will be able to:
- Identify model families and their architectural differences
- Use and compare encoder, decoder, and hybrid models
- Implement practical LLM use cases
- Select appropriate models based on task and resource
- Build and evaluate custom datasets

## 📋 Prerequisites
- Completion of Day 17 (LLMs + Transformers)
- Comfort with APIs, Python async, and JSON
- API keys for OpenAI, Gemini, or similar providers
- Python 3.11+, 16GB RAM+, GPU preferred

---

## 1. Model Architectures and Tokenizers

### Tokenization Approaches
```python
from transformers import AutoTokenizer

# BERT - WordPiece
bert_tokens = AutoTokenizer.from_pretrained('bert-base-uncased').tokenize("ChatGPT is great!")

# GPT - Byte Pair Encoding (BPE)
gpt_tokens = AutoTokenizer.from_pretrained('gpt2').tokenize("ChatGPT is great!")

# T5 - SentencePiece
t5_tokens = AutoTokenizer.from_pretrained('t5-base').tokenize("ChatGPT is great!")
```

### Model Types
- **Encoder (BERT)**: Understanding
- **Decoder (GPT, LLaMA)**: Generation
- **Encoder-Decoder (T5)**: Translation, summarization

```python
# Decoder example
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = "AI is the future"
input_ids = tokenizer(text, return_tensors="pt")
output = model.generate(**input_ids, max_length=40)
print(tokenizer.decode(output[0]))
```

---

## 2. Major Model Families

| Model Family | Type | Key Features | Best For | Python Package |
|--------------|------|--------------|----------|----------------|
| GPT-3.5/4    | Decoder | API-only, strong general performance | Chat, writing, code | `openai` |
| Gemini       | Hybrid  | Multimodal, visual + textual reasoning | Vision + NLP | `google-generativeai` |
| LLaMA 2      | Decoder | Open weights, local deployable | Research, fine-tuning | `llama-cpp-python` |
| Falcon       | Decoder | Efficient, performant open model | Edge, low-latency | `transformers` |

### Example: Comparing APIs
```python
# OpenAI GPT
import openai
openai.api_key = "your-key"
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Explain Python decorators"}]
)
print(response.choices[0].message.content)

# Gemini
import google.generativeai as genai
genai.configure(api_key="your-key")
model = genai.GenerativeModel("gemini-pro")
print(model.generate_content("What are Python decorators?").text)

# Local LLaMA
from llama_cpp import Llama
llm = Llama(model_path="path/to/model.gguf", n_ctx=2048)
print(llm.create_completion("Explain decorators", max_tokens=100)['choices'][0]['text'])
```

---

## 3. Model Selection Guide

### Selection Factors
- **Use Case**:
  - Text generation → GPT, LLaMA
  - Semantic understanding → BERT
  - Translation → T5

- **Resources**:
  - API only → GPT, Gemini
  - Local deploy → LLaMA, Falcon
  - Quantization needed → Falcon or GGUF models

### Decision Flow
```
Start
  ├── API or Local?
  │   ├── API → GPT / Gemini
  │   └── Local → LLaMA / Falcon
  └── Need low compute? → Quantized Falcon
```

---

## 4. Practical Exercises

### Exercise 1: Model Comparison Tool
```python
class ModelEvaluator:
    def __init__(self, models):
        self.models = models

    def evaluate(self, prompt):
        for name, model in self.models.items():
            print(f"--- {name} ---")
            print(model.generate(prompt))
```

### Exercise 2: Custom Dataset Builder
```python
from datasets import Dataset
import pandas as pd

def build_dataset(csv_file):
    df = pd.read_csv(csv_file)
    data = [
        {"instruction": r["question"], "input": r.get("context", ""), "output": r["answer"]}
        for _, r in df.iterrows()
    ]
    return Dataset.from_pandas(pd.DataFrame(data))
```

### Exercise 3: Async LLM API Wrapper
```python
class LLMWrapper:
    async def generate(self, prompt):
        # Call external API here
        return "result"

    async def embed(self, text):
        return [0.1] * 768
```

### Exercise 4: Fine-tuning Setup
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset

def setup_finetune(model_name, dataset_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    dataset = load_dataset(dataset_name)
    return model, tokenizer, dataset
```

---

## ✅ Summary
- Choose model type based on task (encoder vs decoder vs hybrid)
- Evaluate tradeoffs: performance, cost, latency, deployability
- Use APIs for ease; local models for control
- Fine-tune open models for domain-specific intelligence

---

## 🔍 Common Issues and Fixes
| Issue | Solution |
|-------|----------|
| API key error | Store using `os.getenv()` |
| Token length error | Truncate input or raise `max_tokens` |
| CUDA out of memory | Use quantized models or batch processing |

## 📚 Additional Resources
- [HuggingFace Transformers](https://huggingface.co/transformers)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Google Gemini](https://ai.google.dev)
- [LLaMA GitHub](https://github.com/facebookresearch/llama)

## ✅ Knowledge Check
1. What model type is BERT and what tasks is it suited for?
2. How does LLaMA differ from GPT?
3. When should you use a hybrid model like Gemini?
4. List two advantages of using local deployment
5. Why use quantized models?

---

> **Navigation**
> - [← Day 17: LLM + Transformers](17-Python-LLM-Transformers.md)
> - [Day 19: Prompt Engineering →](19-Python-Prompt-Engineering.md)

