# Day 25 – Advanced GPT-4 Programming

## Overview
Master advanced GPT-4 programming techniques, from setup to production-ready implementations. Learn to leverage GPT-4's unique capabilities while following best practices and industry standards.

## Learning Objectives
- Master GPT-4's advanced features and capabilities
- Implement structured prompting patterns
- Build production-ready GPT-4 applications
- Compare and optimize GPT-3.5 vs GPT-4 usage
- Handle complex data extraction and formatting

## Prerequisites
- Strong understanding of LLM concepts
- Experience with API authentication
- Knowledge of async programming
- Familiarity with JSON and data structures
- Understanding of prompt engineering basics

### Technical Requirements
- Python 3.7+
- OpenAI account
- `openai` package
- OpenAI API key

## 1. Advanced GPT-4 Features

### Architecture Overview
```python
from typing import Dict, List

gpt4_capabilities = {
    "context_length": "up to 128k tokens",
    "features": [
        "long_context_understanding",
        "multimodal_input",
        "advanced_reasoning",
        "structured_output"
    ],
    "use_cases": [
        "research_analysis",
        "code_refactoring",
        "ai_assistants",
        "complex_reasoning"
    ]
}
```

## 2. Production Setup

### Configuration Management
```python
from pydantic import BaseSettings
import os

class OpenAIConfig(BaseSettings):
    api_key: str
    org_id: str | None = None
    default_model: str = "gpt-4"
    
    class Config:
        env_prefix = 'OPENAI_'

config = OpenAIConfig()
```

### Client Setup
```python
from openai import OpenAI
from contextlib import asynccontextmanager

@asynccontextmanager
async def get_openai_client():
    client = OpenAI(api_key=config.api_key)
    try:
        yield client
    finally:
        await client.close()
```

## 3. Advanced Prompting Patterns

### Structured Output
```python
async def get_structured_response(prompt: str, output_schema: Dict) -> Dict:
    async with get_openai_client() as client:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": f"Return response in this JSON schema: {output_schema}"
            }, {
                "role": "user",
                "content": prompt
            }],
            response_format={ "type": "json_object" }
        )
        return response.choices[0].message.content
```

### Chain of Thought
```python
async def reasoning_prompt(question: str) -> str:
    async with get_openai_client() as client:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": "Think step by step. Show your reasoning."
            }, {
                "role": "user",
                "content": question
            }]
        )
        return response.choices[0].message.content
```

## 4. Best Practices and Standards

### 🎯 DO's
1. **Use System Messages**
   ```python
   messages = [
       {"role": "system", "content": "You are a precise, technical assistant."},
       {"role": "user", "content": prompt}
   ]
   ```

2. **Implement Rate Limiting**
   ```python
   import asyncio
   from typing import List
   
   async def batch_process(prompts: List[str], delay: float = 0.1):
       results = []
       for prompt in prompts:
           result = await generate_text(prompt)
           results.append(result)
           await asyncio.sleep(delay)
       return results
   ```

3. **Handle Errors Properly**
   ```python
   from openai import OpenAIError
   
   async def safe_generate(prompt: str):
       try:
           async with get_openai_client() as client:
               response = await client.chat.completions.create(
                   model="gpt-4",
                   messages=[{"role": "user", "content": prompt}]
               )
               return response.choices[0].message.content, None
       except OpenAIError as e:
           return None, f"API Error: {str(e)}"
   ```

### ❌ DON'Ts
1. **Avoid Token Waste**
   ```python
   # BAD
   prompt = "Write a very long essay about..."
   
   # GOOD
   prompt = "Write a concise summary focusing on key points about..."
   ```

2. **Don't Skip Validation**
   ```python
   from pydantic import BaseModel
   
   class PromptRequest(BaseModel):
       content: str
       max_tokens: int = 1000
       temperature: float = 0.7
       
       class Config:
           min_content_length = 10
           max_content_length = 4000
   ```

### 🔧 Standard Practices
1. **Model Selection Logic**
   ```python
   def select_model(task_type: str, complexity: int) -> str:
       if complexity > 7 or task_type in ["reasoning", "code"]:
           return "gpt-4"
       return "gpt-3.5-turbo"
   ```

2. **Response Processing**
   ```python
   from typing import TypedDict
   
   class GPTResponse(TypedDict):
       content: str
       tokens_used: int
       cost_estimate: float
       
   def process_response(response) -> GPTResponse:
       return {
           "content": response.choices[0].message.content,
           "tokens_used": response.usage.total_tokens,
           "cost_estimate": calculate_cost(response.usage.total_tokens)
       }
   ```

## ✅ Assignments

### Level 1: Basic Implementation
1. Create a GPT-4 wrapper class with proper error handling
2. Implement structured output extraction
3. Compare responses between GPT-3.5 and GPT-4

### Level 2: Advanced Features
1. Build a prompt template system
2. Create a cost tracking mechanism
3. Implement response validation

### Bonus Challenge
- Create a FastAPI service using GPT-4
- Add comprehensive logging
- Implement rate limiting and caching

## 🧠 Summary
- GPT-4 excels at complex reasoning tasks
- Proper setup and error handling are crucial
- Cost and token management are important
- Structured outputs improve reliability

> **Navigation**
> - [← GPT-4 API](24-Python-GPT4-API.md)
> - [LLM Chains →](26-Python-LLM-Chains-FastAPI.md)
