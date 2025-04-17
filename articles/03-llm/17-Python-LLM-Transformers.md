# Day 17 - Introduction to LLMs and Transformer Architecture

## Overview
Explore the foundation of Large Language Models (LLMs) and the transformer architecture behind models like GPT, BERT, and T5. This session dives deep into tokenization, embeddings, attention mechanisms, and real-world LLM applications.

## 🌟 Learning Objectives
By the end of this lesson, you will be able to:
- Understand core components of LLMs
- Grasp how tokenization and embeddings work
- Differentiate between encoder, decoder, and encoder-decoder models
- Analyze attention mechanisms (self, cross, masked)
- Use transformers via HuggingFace pipelines
- Calculate token usage and optimize generation

## 📋 Prerequisites
- Solid understanding of Python and deep learning basics
- Experience with neural networks, matrix math, and NLP concepts
- Python 3.11+ installed
- GPU recommended for training

---

## 1. Theoretical Foundation

### What is an LLM?
Large Language Models (LLMs) are deep learning models trained on massive corpora to generate, understand, and reason with human-like text.

- GPT-3: 175B parameters
- GPT-4: >1T parameters (estimated)
- Claude 2: 100K+ token context window

### Tokenization
```python
from transformers import GPT2Tokenizer

text = "Python programming is fun!"
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokens = tokenizer.encode(text)

print(f"Tokens: {tokens}")
print(f"Token count: {len(tokens)}")
```

### Context Window Limits
| Model      | Max Tokens |
|------------|------------|
| GPT-3      | 4,096      |
| GPT-4      | 8,192+     |
| Claude 2   | 100,000    |

---

## 2. Architecture Overview

### Encoder (BERT)
Bi-directional reading of text input.
```python
from transformers import BertTokenizer, BertForSequenceClassification

text = "This movie is fantastic!"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer(text, return_tensors="pt")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
output = model(**tokens)
print(output.logits.softmax(dim=1))
```

### Decoder (GPT)
Generates text auto-regressively.
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0]))
```

### Encoder-Decoder (T5, MarianMT)
```python
from transformers import MarianMTModel, MarianTokenizer

model_name = 'Helsinki-NLP/opus-mt-en-fr'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = "How are you?"
translated = model.generate(**tokenizer(text, return_tensors="pt"))
print(tokenizer.decode(translated[0], skip_special_tokens=True))
```

---

## 3. Core Components of Transformer

### Embedding Layer
```python
vocab_size = 50257
embedding_dim = 768
sequence_length = 1024
```
- Includes positional encodings
- Converts tokens into vector space

### Self-Attention
```python
attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
attention_probs = torch.softmax(attention_scores, dim=-1)
attention_output = torch.matmul(attention_probs, V)
```

### Feed Forward Network
- Linear → ReLU → Linear

### Residual + LayerNorm
- Add skip connections and normalize

---

## 4. Attention Mechanisms

### Multi-Head Attention
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

    def forward(self, x):
        B = x.size(0)
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, -1, self.num_heads * self.head_dim)
        return self.proj(out)
```

### Patterns:
- **Self-Attention**: Within sequence
- **Cross-Attention**: Between sequences (encoder-decoder)
- **Masked Attention**: Prevents future token leakage in decoders

---

## 5. Practical Use Cases (with Pipelines)

### Text Generation
```python
from transformers import pipeline
gen = pipeline("text-generation", model="gpt2")
print(gen("AI will soon", max_length=50)[0]['generated_text'])
```

### Question Answering
```python
qa = pipeline("question-answering")
context = "Python is a high-level programming language."
question = "What is Python?"
print(qa(question=question, context=context)['answer'])
```

### Sentiment Analysis
```python
classifier = pipeline("sentiment-analysis")
print(classifier("I love this product!"))
```

### Translation
```python
translator = pipeline("translation_en_to_fr")
print(translator("Hello, how are you?")[0]['translation_text'])
```

---

## 6. Setup & Installation

### Requirements File
```txt
transformers>=4.30.0
torch>=2.0.0
tokenizers>=0.13.0
datasets>=2.12.0
accelerate>=0.20.0
sentencepiece>=0.1.99
sacremoses>=0.0.53
protobuf>=3.20.0
einops>=0.6.1
bitsandbytes>=0.41.0
```

### Install Steps
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 7. Popular LLM Models

| Model     | Org         | Params   | Highlights            | Package           |
|-----------|-------------|----------|------------------------|--------------------|
| GPT-4     | OpenAI      | ?        | Reasoning, API only   | `openai`          |
| BERT      | Google      | 110M-340M| Bidirectional encoder | `transformers`    |
| LLaMA 2   | Meta        | 7B-70B   | Open weights          | `transformers`    |
| Claude 2  | Anthropic   | ?        | 100K token context    | `anthropic`       |
| Gemini    | Google      | ?        | Multimodal            | `google-generativeai` |

---

## 8. Hands-on Practice

### Exercise 1: Count Tokens
```python
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = input("Enter text: ")
tokens = tokenizer.encode(text)
print(f"Token count: {len(tokens)}")
```

### Exercise 2: Generate Text with Prompt
```python
from transformers import pipeline
gen = pipeline("text-generation", model="gpt2")
print(gen("AI is the future of", max_length=30)[0]['generated_text'])
```

### Exercise 3: Sentiment Analysis Tool
```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
text = input("Enter review: ")
print(classifier(text))
```

---

## ✅ Summary
- LLMs use token embeddings and attention
- Transformers consist of encoders, decoders, or both
- Attention mechanisms drive contextual understanding
- HuggingFace makes it easy to use LLMs in Python
- Optimize for token efficiency and model suitability

---

## 🔍 Common Issues and Fixes
| Issue | Solution |
|-------|----------|
| Token mismatch | Use same tokenizer as model |
| CUDA errors | Check GPU availability and memory |
| API key missing | Set env variable or config for OpenAI, Anthropic |

## 📚 Additional Resources
- [HuggingFace Transformers](https://huggingface.co/transformers)
- [Attention Is All You Need – Paper](https://arxiv.org/abs/1706.03762)
- [BERT Explained](https://jalammar.github.io/illustrated-bert/)
- [OpenAI Tokenizer Tool](https://platform.openai.com/tokenizer)

## ✅ Knowledge Check
1. What is the role of the embedding layer?
2. What is the difference between encoder and decoder?
3. When should you use masked attention?
4. List three real-world use cases of transformers.
5. What does multi-head attention do?

---

> **Navigation**
> - [← FastAPI Mini Project](15-Python-FastAPI-Mini-Project.md)
> - [Day 18: LLM Applications →](18-Python-LLM-Applications.md)

