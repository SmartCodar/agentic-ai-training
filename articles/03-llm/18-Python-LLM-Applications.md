# Day 18 - Model Families and LLM Applications

## Overview
Learn about different LLM model families, their architectures, and how to use them in practical Python applications.

## Learning Objectives
- Understand different LLM architectures and their use cases
- Learn to use major model families (GPT, BERT, LLaMA, Gemini)
- Implement practical LLM applications
- Compare model performance and requirements
- Make informed model selection decisions

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
- Completion of [17 - LLM Transformers](17-Python-LLM-Transformers.md)

### Technical Requirements
- Python 3.7+
- GPU recommended for model inference
- 16GB+ RAM recommended
- Internet connection for API access

## Time Estimate
- Reading: 45 minutes
- Practice: 60 minutes
- Exercises: 45 minutes

---

## 1. Model Architectures and Types

### Tokenizer Types
A tokenizer converts text into numerical tokens. Different models use different approaches:

```python
from transformers import AutoTokenizer

# 1. WordPiece (BERT)
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_tokens = bert_tokenizer.tokenize("ChatGPT is great!")
print(f"BERT tokens: {bert_tokens}")

# 2. BPE (GPT)
gpt_tokenizer = AutoTokenizer.from_pretrained('gpt2')
gpt_tokens = gpt_tokenizer.tokenize("ChatGPT is great!")
print(f"GPT tokens: {gpt_tokens}")

# 3. SentencePiece (T5)
t5_tokenizer = AutoTokenizer.from_pretrained('t5-base')
t5_tokens = t5_tokenizer.tokenize("ChatGPT is great!")
print(f"T5 tokens: {t5_tokens}")
```

### Model Types and Use Cases

1. **Encoder Models (BERT)**
```python
from transformers import BertModel, BertTokenizer
import torch

# Load model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Get embeddings
text = "Understanding language models"
inputs = tokenizer(text, return_tensors="pt", padding=True)
outputs = model(**inputs)
embeddings = outputs.last_hidden_state
print(f"Embeddings shape: {embeddings.shape}")
```

2. **Decoder Models (GPT)**
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Generate text
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated: {generated_text}")
```

3. **Encoder-Decoder Models (T5)**
```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Translate text
text = "translate English to French: Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs)
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Translation: {translation}")
```

## 2. Major Model Families

| Model Family | Type | Key Features | Best For | Python Package |
|-------------|------|--------------|----------|----------------|
| GPT-4/3.5 | Decoder | - Strong general knowledge<br>- Good at following instructions<br>- API access only | - Chat applications<br>- Content generation<br>- Code assistance | `openai` |
| Gemini | Hybrid | - Multimodal capabilities<br>- Strong reasoning<br>- API or local deployment | - Vision tasks<br>- Complex reasoning<br>- Multi-step problems | `google-generativeai` |
| LLaMA 2 | Decoder | - Open weights<br>- Multiple sizes (7B-70B)<br>- Fine-tunable | - Custom applications<br>- Local deployment<br>- Research | `llama-cpp-python` |
| Falcon | Decoder | - Efficient architecture<br>- Strong performance<br>- Open weights | - Resource-constrained<br>- Edge deployment<br>- Fine-tuning | `transformers` |

### Example: Using Multiple Models

```python
# 1. OpenAI GPT
import openai

openai.api_key = 'your-api-key'
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Explain Python generators"}]
)
print(response.choices[0].message.content)

# 2. Google Gemini
import google.generativeai as genai

genai.configure(api_key='your-api-key')
model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("Explain Python generators")
print(response.text)

# 3. Local LLaMA
from llama_cpp import Llama

llm = Llama(
    model_path="path/to/llama-2-7b.gguf",
    n_ctx=2048,
    n_threads=4
)
response = llm.create_completion(
    "Explain Python generators",
    max_tokens=100,
    temperature=0.7
)
print(response['choices'][0]['text'])
```

## 3. Model Selection Guide

### Factors to Consider
1. **Task Requirements**
   - Text generation → Decoder models (GPT, LLaMA)
   - Understanding → Encoder models (BERT)
   - Translation → Encoder-Decoder models (T5)

2. **Resource Constraints**
   - Limited compute → Smaller models or API
   - Need privacy → Local deployment
   - Cost sensitive → Open source models

3. **Performance Needs**
   - Speed → Optimized models (Falcon)
   - Accuracy → Larger models
   - Specific domain → Fine-tuned models

### Decision Flowchart
```
Start
  ├── Need API access?
  │   ├── Yes → GPT/Gemini
  │   └── No → Continue
  ├── Have GPU?
  │   ├── Yes → LLaMA/Falcon
  │   └── No → Continue
  ├── Memory limited?
  │   ├── Yes → Quantized models
  │   └── No → Full models
  └── End
```

## Summary

### Key Takeaways
1. Different architectures suit different tasks
2. Model selection depends on resources and requirements
3. APIs provide easy access but local deployment offers control
4. Open source models enable customization

### Best Practices
1. Start with small models and scale up
2. Use API for prototyping
3. Consider fine-tuning for specific tasks
4. Monitor resource usage and costs

## 4. Practical Exercises

### Exercise 1: Model Comparison Pipeline
```python
from typing import Dict, List
import time

class ModelEvaluator:
    def __init__(self, models: Dict):
        self.models = models
        self.results = {}

    def evaluate_model(self, model_name: str, test_cases: List[str]):
        model = self.models[model_name]
        results = {
            'latency': [],
            'token_count': [],
            'responses': []
        }

        for test in test_cases:
            start_time = time.time()
            response = model.generate(test)  # Implement for your model
            latency = time.time() - start_time
            
            results['latency'].append(latency)
            results['responses'].append(response)
            # Add token counting logic

        return results

# Usage example
test_cases = [
    "Explain Python decorators",
    "What is dependency injection?",
    "Compare SQL vs NoSQL"
]

# Initialize models (implement for your chosen models)
models = {
    'gpt': OpenAIModel(),
    'llama': LlamaModel(),
    'bert': BertModel()
}

evaluator = ModelEvaluator(models)
results = evaluator.evaluate_model('gpt', test_cases)
```

### Exercise 2: Custom Dataset Creation
```python
from datasets import Dataset
import pandas as pd

def create_instruction_dataset(data_source: str, output_file: str):
    """Create a dataset for instruction tuning"""
    # 1. Load and preprocess data
    df = pd.read_csv(data_source)
    
    # 2. Format as instruction-response pairs
    formatted_data = [{
        'instruction': row['question'],
        'input': row.get('context', ''),  # Optional context
        'output': row['answer']
    } for _, row in df.iterrows()]
    
    # 3. Create HuggingFace dataset
    dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
    
    # 4. Save dataset
    dataset.save_to_disk(output_file)
    return dataset

# Example usage
dataset = create_instruction_dataset(
    'questions.csv',
    'instruction_dataset'
)
```

### Exercise 3: Model API Wrapper
```python
from abc import ABC, abstractmethod
from typing import Optional, Dict
import asyncio

class LLMBase(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    async def embed(self, text: str) -> list:
        pass

class OpenAIWrapper(LLMBase):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    async def generate(self, prompt: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            return ""

    async def embed(self, text: str) -> list:
        try:
            response = await self.client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error: {e}")
            return []

# Usage
async def main():
    llm = OpenAIWrapper(api_key="your-key")
    result = await llm.generate("Explain async in Python")
    print(result)

# Run with: asyncio.run(main())
```

### Exercise 4: Fine-tuning Pipeline
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset

def prepare_fine_tuning(base_model: str, dataset_path: str, output_dir: str):
    # 1. Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # 2. Load and preprocess dataset
    dataset = load_dataset(dataset_path)
    
    def preprocess(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=512
        )

    processed_dataset = dataset.map(preprocess, batched=True)

    # 3. Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=1000,
        logging_steps=100,
    )

    return model, tokenizer, processed_dataset, training_args
```

### Best Practices and Guidelines

#### DO's:
1. **Data Preparation**
   - Clean and validate input data
   - Use appropriate tokenization
   - Balance your datasets
   - Include diverse examples

2. **Model Usage**
   - Start with smaller models first
   - Use batching for multiple inputs
   - Implement proper error handling
   - Monitor resource usage

3. **API Integration**
   - Use async for better performance
   - Implement rate limiting
   - Handle API errors gracefully
   - Cache results when appropriate

4. **Deployment**
   - Use model quantization
   - Implement proper logging
   - Monitor model performance
   - Have fallback options

#### DON'Ts:
1. **Data Handling**
   ❌ Don't use unprocessed raw text
   ❌ Don't ignore token limits
   ❌ Don't skip data validation
   ❌ Don't use biased training data

2. **Model Usage**
   ❌ Don't load multiple large models simultaneously
   ❌ Don't ignore memory constraints
   ❌ Don't use synchronous calls for multiple requests
   ❌ Don't hardcode model parameters

3. **API Usage**
   ❌ Don't expose API keys in code
   ❌ Don't ignore rate limits
   ❌ Don't skip error handling
   ❌ Don't make unnecessary API calls

4. **Deployment**
   ❌ Don't deploy without monitoring
   ❌ Don't ignore resource limits
   ❌ Don't skip security measures
   ❌ Don't use production keys in development

## 5. Structured Learning Exercises

### Level 1: Fundamentals (2-3 hours)

#### Exercise 1.1: Understanding Core Concepts
```python
# Create a simple tokenizer demonstration
from transformers import AutoTokenizer

def demonstrate_tokenization():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    text = "Let's understand how tokenization works!"
    tokens = tokenizer.tokenize(text)
    print(f"Original text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}")

# Run this to see tokenization in action
demonstrate_tokenization()
```

#### Tasks:
1. **Define and Demonstrate Key Components**
   - Implement tokenization examples for different models
   - Show encoder vs decoder differences with code
   - Calculate and display context window usage
   - Compare token counts across models

2. **Model Family Classification**
   Create a classification function:
   ```python
   def classify_model(model_name: str) -> str:
       classifications = {
           'bert': 'Encoder',
           'gpt': 'Decoder',
           't5': 'Encoder-Decoder',
           'llama': 'Decoder'
       }
       return classifications.get(model_name.lower(), 'Unknown')
   ```

### Level 2: Intermediate Applications (4-6 hours)

#### Exercise 2.1: Model Comparison Framework
```python
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ModelMetrics:
    name: str
    provider: str
    architecture: str
    parameters: str
    token_limit: int
    license: str
    features: List[str]

def compare_models(models: List[ModelMetrics]) -> Dict:
    comparison = {}
    for model in models:
        comparison[model.name] = {
            'Provider': model.provider,
            'Architecture': model.architecture,
            'Parameters': model.parameters,
            'Token Limit': model.token_limit,
            'License': model.license,
            'Key Features': model.features
        }
    return comparison

# Example usage
models = [
    ModelMetrics(
        name="GPT-4",
        provider="OpenAI",
        architecture="Decoder",
        parameters="Unknown",
        token_limit=128000,
        license="Commercial",
        features=["Text generation", "Code completion", "Chat"]
    ),
    # Add more models...
]
```

#### Exercise 2.2: Email Summarization System
```python
from typing import List
import asyncio

class EmailSummarizer:
    def __init__(self, model_name: str, batch_size: int = 10):
        self.model_name = model_name
        self.batch_size = batch_size
        self.total_tokens = 0
        self.processing_time = 0

    async def summarize_batch(self, emails: List[str]) -> List[str]:
        # Implement batched summarization
        pass

    async def process_emails(self, emails: List[str]) -> Dict:
        results = {
            'summaries': [],
            'metrics': {
                'total_emails': len(emails),
                'total_tokens': 0,
                'processing_time': 0
            }
        }
        # Implement email processing
        return results
```

### Level 3: Advanced Challenges (8+ hours)

#### Challenge 1: Open Source Model Analysis
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelLicense:
    name: str
    allows_commercial: bool
    requires_attribution: bool
    allows_modification: bool
    share_alike: bool

@dataclass
class OpenSourceModel:
    name: str
    weights_available: bool
    code_available: bool
    tokenizer_available: bool
    license: ModelLicense
    training_data_available: Optional[bool] = False

def analyze_model_openness(model: OpenSourceModel) -> Dict:
    score = 0
    reasons = []
    
    if model.weights_available:
        score += 40
        reasons.append("Weights are publicly available")
    
    if model.code_available:
        score += 30
        reasons.append("Source code is available")
    
    if model.tokenizer_available:
        score += 20
        reasons.append("Tokenizer is available")
    
    if model.training_data_available:
        score += 10
        reasons.append("Training data is available")
    
    return {
        'openness_score': score,
        'reasons': reasons,
        'license_details': vars(model.license)
    }
```

#### Challenge 2: LLaMA Deployment Pipeline
```python
from pathlib import Path
import torch
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class DeploymentConfig:
    model_size: str  # '7B', '13B', '70B'
    quantization: str  # '4-bit', '8-bit', 'none'
    max_batch_size: int
    gpu_memory: int  # GB
    cpu_threads: int

class LLaMADeployer:
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def check_requirements(self) -> Dict[str, bool]:
        return {
            'gpu_available': torch.cuda.is_available(),
            'memory_sufficient': self._check_memory(),
            'disk_space_sufficient': self._check_disk_space(),
            'cpu_threads_available': self._check_cpu_threads()
        }

    def estimate_resources(self) -> Dict:
        return {
            'gpu_memory_required': self._calculate_gpu_memory(),
            'cpu_memory_required': self._calculate_cpu_memory(),
            'disk_space_required': self._calculate_disk_space(),
            'estimated_throughput': self._estimate_throughput()
        }

    def deploy(self) -> bool:
        # Implement deployment steps
        pass
```

### Evaluation Criteria

#### Level 1 (Fundamentals)
- Understanding of tokenization ✓
- Correct model classification ✓
- Basic code implementation ✓
- Documentation quality ✓

#### Level 2 (Intermediate)
- Framework design ✓
- Error handling ✓
- Performance optimization ✓
- Testing approach ✓

#### Level 3 (Advanced)
- System architecture ✓
- Resource management ✓
- Scalability considerations ✓
- Production readiness ✓

### Submission Guidelines
1. Code must be properly documented
2. Include unit tests
3. Provide performance metrics
4. Add deployment instructions

---

> **Navigation**
> - [← LLM Transformers](17-Python-LLM-Transformers.md)
> - [Prompt Engineering →](19-Python-Prompt-Engineering.md)
