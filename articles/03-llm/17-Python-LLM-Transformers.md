# Day - 17 Introduction to LLMs and Transformer Architecture

## Overview
Learn the fundamentals of Large Language Models (LLMs) and the Transformer architecture that powers modern AI systems like GPT, BERT, and T5.

## Learning Objectives
- Master core LLM concepts and components
- Understand tokenization and embeddings
- Learn encoder and decoder architectures
- Study attention mechanisms in detail
- Implement practical LLM applications
- Calculate and optimize token usage

## Prerequisites
- Completion of [01 - Python Basics](../01-basics/01-Python-Basics-Variables-Types-Operators.md)
- Completion of [02 - Flow Control](../01-basics/02-Python-Flow-Control-Loops-Conditions.md)
- Completion of [03 - Functions](../01-basics/03-Python-Functions-Modular-Programming.md)
- Completion of [04 - Modules and Packages](../01-basics/04-Python-Modules-Packages.md)
- Completion of [05 - Object-Oriented Programming](../01-basics/05-Python-OOP.md)
- Completion of [06 - File Handling](../01-basics/06-Python-File-Handling.md)
- Completion of [07 - Testing and Debugging](../01-basics/07-Python-Testing-Debugging.md)
- Completion of [08 - Functional Programming](../01-basics/08-Python-Functional-Programming.md)
- Completion of [09 - Project Setup](../01-basics/09-Python-Project-Setup.md)
- Completion of [10 - Async Programming](../02-fastapi/10-Python-Async-Programming.md)
- Completion of [11 - Aiohttp Client](../02-fastapi/11-Python-Aiohttp-Client.md)
- Completion of [12 - FastAPI Basics](../02-fastapi/12-Python-FastAPI.md)
- Completion of [13 - FastAPI Routes](../02-fastapi/13-Python-FastAPI-Routes.md)
- Completion of [14 - Pydantic](../02-fastapi/14-Python-Pydantic.md)
- Completion of [15 - FastAPI Mini Project](../02-fastapi/15-Python-FastAPI-Mini-Project.md)
- Completion of [16 - FastAPI Project](../02-fastapi/16-Python-FastAPI-Project.md)

### Technical Requirements
- Python 3.7+
- Basic understanding of deep learning concepts
- Familiarity with matrix operations
- Understanding of natural language processing basics

## Time Estimate
- Reading: 45 minutes
- Practice: 30 minutes
- Exercises: 45 minutes

---

## 1. Core LLM Concepts

### What is an LLM?
A Large Language Model (LLM) is a deep learning model trained on massive amounts of text data to understand, generate, and reason with human language. Modern LLMs like GPT-4 have hundreds of billions of parameters and can perform a wide range of tasks.

### Tokenization
Tokenization is the process of converting text into numerical tokens that the model can process.

#### Example of Tokenization:
```python
# Original text
text = "Hello, how are you?"

# Tokenized (simplified)
tokens = ["Hello", ",", "how", "are", "you", "?"]

# Token IDs (numerical representation)
token_ids = [15496, 11, 2129, 2024, 2017, 30]
```

#### Token Calculation:
```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text = "Python programming is fun!"
tokens = tokenizer.encode(text)

print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token count: {len(tokens)}")
```

### Context Window
The context window is the maximum number of tokens a model can process at once.

Example sizes:
- GPT-3: 4,096 tokens
- GPT-4: 8,192 tokens
- Claude 2: 100,000 tokens

### Parameters
Parameters are the learned weights in the model that capture patterns from training data.

Model sizes:
- GPT-3: 175 billion parameters
- PaLM: 540 billion parameters
- GPT-4: estimated >1 trillion parameters

## 2. Transformer Architecture

### Encoder
The encoder processes input text bi-directionally (can look at both past and future tokens).

#### Example Encoder Use Case - Sentiment Analysis:
```python
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

text = "This movie is fantastic!"
tokens = tokenizer(text, return_tensors="pt")
output = model(**tokens)

# Output: positive sentiment score
print(f"Sentiment score: {output.logits.softmax(dim=1)}")
```

### Decoder
The decoder generates text auto-regressively (can only look at past tokens).

#### Example Decoder Use Case - Text Generation:
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Generate continuation
outputs = model.generate(
    input_ids,
    max_length=50,
    temperature=0.7,
    num_return_sequences=1
)

print(tokenizer.decode(outputs[0]))
```

### Encoder-Decoder (Sequence-to-Sequence)
Combines both architectures for tasks like translation.

#### Example Translation:
```python
from transformers import MarianMTModel, MarianTokenizer

model_name = 'Helsinki-NLP/opus-mt-en-fr'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = "Hello, how are you?"
translated = model.generate(**tokenizer(text, return_tensors="pt"))

print(tokenizer.decode(translated[0], skip_special_tokens=True))
```

### Key Components:
1. **Embedding Layer**:
   - Converts tokens to vectors (typically 768 dimensions)
   - Includes positional encoding
   ```python
   # Example embedding dimensions
   vocab_size = 50257  # GPT-2 vocabulary size
   embedding_dim = 768
   sequence_length = 1024
   ```

2. **Self-Attention Layer**:
   - Calculates attention scores between all tokens
   - Uses Query (Q), Key (K), and Value (V) matrices
   ```python
   # Simplified attention calculation
   attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
   attention_probs = torch.softmax(attention_scores, dim=-1)
   attention_output = torch.matmul(attention_probs, V)
   ```

3. **Feed-Forward Network (FFN)**:
   - Processes attention outputs
   - Typically has two linear layers with ReLU activation

4. **Layer Normalization + Residuals**:
   - Stabilizes training
   - Helps with gradient flow
## 3. Attention Mechanisms

### Multi-Head Attention
Allows the model to focus on different aspects of the input simultaneously.

#### Example Implementation:
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size = x.size(0)
        qkv = self.qkv(x)
        
        # Split into heads
        qkv = qkv.reshape(batch_size, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        return self.proj(out)
```

### Attention Patterns
1. **Self-Attention**: Tokens attend to other tokens in the same sequence
2. **Cross-Attention**: Tokens attend to tokens from another sequence
3. **Masked Attention**: Used in decoders to prevent looking at future tokens

## 4. Practical Applications

### 1. Text Generation
```python
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')
text = generator("The future of AI is", max_length=50)[0]['generated_text']
print(text)
```

### 2. Question Answering
```python
from transformers import pipeline

qa = pipeline('question-answering')
context = "Python is a high-level programming language known for its simplicity."
question = "What is Python known for?"

answer = qa(question=question, context=context)
print(f"Answer: {answer['answer']}")
```

### 3. Text Classification
```python
from transformers import pipeline

classifier = pipeline('sentiment-analysis')
result = classifier("This new feature is amazing!")
print(f"Sentiment: {result[0]['label']}")
```

### 4. Translation
```python
from transformers import pipeline

translator = pipeline('translation_en_to_fr')
result = translator("Hello, how are you?")
print(f"Translation: {result[0]['translation_text']}")
```

## Summary

### Key Takeaways
1. LLMs process text using tokens and embeddings
2. Transformer architecture uses encoders and decoders
3. Attention mechanisms enable context understanding
4. Models can be fine-tuned for specific tasks

### Best Practices
1. Choose appropriate model size for your task
2. Consider context window limitations
3. Optimize token usage
4. Use appropriate temperature for generation
5. Implement proper error handling

### Next Steps
1. Experiment with different models
2. Try fine-tuning for specific tasks
3. Build practical applications
4. Optimize for production use

## Requirements and Setup

### Dependencies
Create a `requirements.txt` file with the following:
```txt
# Core dependencies
torch>=2.0.0
transformers>=4.30.0
tokenizers>=0.13.0
datasets>=2.12.0
accelerate>=0.20.0

# Optional but recommended
sentencepiece>=0.1.99  # For some tokenizers
sacremoses>=0.0.53    # For some models
protobuf>=3.20.0      # Required by some models
einops>=0.6.1         # For attention operations
bitsandbytes>=0.41.0  # For 4-bit quantization
```

### Installation
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Popular Models

| Model    | Organization | Parameters | Key Features          | Python Package |
|----------|-------------|------------|----------------------|----------------|
| GPT-4    | OpenAI      | Unknown    | Multimodal, reasoning | `openai` |
| LLaMA 2  | Meta        | 7B-70B     | Open weights         | `llama` |
| BERT     | Google      | 110M-340M  | Bidirectional        | `transformers` |
| Gemini   | Google      | Unknown    | Multimodal AI        | `google-generativeai` |
| Claude 2 | Anthropic   | Unknown    | Long context         | `anthropic` |

### Example Model Usage
```python
# 1. Using BERT
from transformers import AutoModel, AutoTokenizer

model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. Using GPT-2 (as GPT-4 requires API key)
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 3. Using T5 for translation
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
```

---

> **Navigation**
> - [← FastAPI Project](../02-fastapi/16-Python-FastAPI-Project.md)
> - [LLM Applications →](18-Python-LLM-Applications.md)

## 2. Core Concepts

### Tokens
```python
# Example of tokenization
text = "Learning about LLMs is fascinating!"
tokens = ["Learn", "ing", " about", " LLM", "s", " is", " fascin", "ating", "!"]

# Each token is converted to a numerical ID
token_ids = [234, 456, 789, 567, 34, 89, 678, 345, 12]
```

### Context Window
- Maximum sequence length the model can process
- Determines how much "memory" the model has
- Examples:
  - GPT-3: 4K tokens
  - GPT-4: 32K tokens
  - Claude 2: 100K tokens

### Parameters
- Learned weights during training
- More parameters generally mean more capabilities
- Examples of parameter counts:
  ```python
  models = {
      "GPT-3": "175 billion",
      "LLaMA 2": "70 billion",
      "BERT-Large": "340 million",
      "GPT-2": "1.5 billion"
  }
  ```

## 3. Transformer Architecture

### High-Level Overview
```plaintext
Input Text → Tokenization → Embeddings → Transformer Blocks → Output
                                            ↓
                              [Self-Attention + Feed Forward]
```

### Key Components
1. **Embeddings**
   - Convert tokens to vectors
   - Capture semantic meaning
   - Include positional information

2. **Self-Attention**
   - Allows words to "look at" other words
   - Computes importance scores
   - Parallel processing

3. **Feed-Forward Networks**
   - Process attended information
   - Apply non-linear transformations
   - Project to output space

### Attention Mechanism
```python
def self_attention(query, key, value):
    # Compute attention scores
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(key.size(-1))
    
    # Apply softmax for probability distribution
    attention_weights = F.softmax(scores, dim=-1)
    
    # Compute weighted sum of values
    return attention_weights @ value
```

## 4. Real-World Applications

### 1. Text Generation
```python
from transformers import pipeline

generator = pipeline('text-generation')
prompt = "The future of AI is"
response = generator(prompt, max_length=50)
```

### 2. Question Answering
```python
qa_model = pipeline('question-answering')
context = "LLMs are trained on vast amounts of text data."
question = "What are LLMs trained on?"
answer = qa_model(question=question, context=context)
```

### 3. Code Generation
```python
# Example using GitHub Copilot or similar
def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number"""
    # LLM can generate the implementation
```

## 5. Best Practices

### When Using LLMs
1. Always validate outputs
2. Consider ethical implications
3. Handle sensitive data carefully
4. Monitor token usage
5. Implement proper error handling

### Common Pitfalls
1. Overreliance on model outputs
2. Ignoring context limitations
3. Not handling edge cases
4. Poor prompt engineering
5. Insufficient output validation

## Summary

### Key Takeaways
1. LLMs are powerful language models
2. Transformer architecture is fundamental
3. Attention mechanism is crucial
4. Real-world applications are diverse
5. Best practices ensure reliable use

### Next Steps
1. Experiment with different models
2. Practice prompt engineering
3. Build practical applications
4. Study advanced concepts

---

> **Navigation**
> - [← FastAPI Mini Project](15-Python-FastAPI-Mini-Project.md)
> - [LLM Applications →](18-Python-LLM-Applications.md)
