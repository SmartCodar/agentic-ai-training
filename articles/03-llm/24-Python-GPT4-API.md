# Day 24 – Introduction to GPT-4 API

## Overview
Master OpenAI's GPT-4 API integration, from setup to advanced usage. Learn to leverage GPT-4's capabilities for various applications while following best practices and standards.

## Learning Objectives
- Understand GPT-4's architecture and capabilities
- Set up and configure OpenAI API access
- Write effective prompts for GPT-4
- Implement function calling and tools
- Build production-ready applications

## Prerequisites
- Strong understanding of LLM concepts
- Experience with API authentication and security
- Knowledge of prompt engineering principles
- Familiarity with async programming
- Understanding of JSON and structured data

### Technical Requirements
- Python 3.7+
- OpenAI account
- `openai` package installed
- OpenAI API key

## 1. Understanding GPT-4

### Architecture and Capabilities
GPT-4 is OpenAI's most advanced LLM:
- Multimodal capabilities (text and vision)
- Advanced reasoning and tool usage
- Function calling support
- Extensive context window (up to 128k tokens)

### Key Features
```python
# Key capabilities of GPT-4
capabilities = {
    "modalities": ["text", "vision"],
    "context_length": "up to 128k tokens",
    "strengths": [
        "reasoning",
        "function_calling",
        "tool_usage",
        "code_generation"
    ]
}
```

## 2. Setting Up GPT-4 Access

### API Configuration
```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
```

### Basic Usage
```python
async def generate_text(prompt: str) -> str:
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    return response.choices[0].message.content
```

## 3. Advanced Features

### Function Calling
```python
from typing import List, Dict

def define_functions() -> List[Dict]:
    return [{
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }
    }]

async def chat_with_functions(prompt: str):
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        functions=define_functions()
    )
    return response
```

### Vision API
```python
async def analyze_image(image_url: str, prompt: str):
    response = await client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": image_url}
            ]
        }]
    )
    return response.choices[0].message.content
```

## 4. Best Practices and Standards

### 🎯 DO's
1. **API Key Security**
   ```python
   from pydantic import BaseSettings
   
   class OpenAIConfig(BaseSettings):
       api_key: str
       org_id: str | None = None
       
       class Config:
           env_prefix = 'OPENAI_'
   ```

2. **Rate Limiting**
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

3. **Error Handling**
   ```python
   from openai import OpenAIError
   
   async def safe_generate(prompt: str):
       try:
           return await generate_text(prompt), None
       except OpenAIError as e:
           return None, f"API Error: {str(e)}"
       except Exception as e:
           return None, f"Unexpected error: {str(e)}"
   ```

4. **Response Validation**
   ```python
   from pydantic import BaseModel
   
   class GPTResponse(BaseModel):
       content: str
       tokens_used: int
       finish_reason: str
   
   def validate_response(response) -> GPTResponse:
       return GPTResponse(
           content=response.choices[0].message.content,
           tokens_used=response.usage.total_tokens,
           finish_reason=response.choices[0].finish_reason
       )
   ```

### ❌ DON'Ts
1. **Never Expose API Keys**
   ```python
   # BAD
   api_key = "sk-..."  # Never hardcode
   
   # GOOD
   api_key = os.getenv('OPENAI_API_KEY')
   ```

2. **Don't Ignore Rate Limits**
   ```python
   # BAD
   results = [generate_text(p) for p in prompts]  # No delay
   
   # GOOD
   results = await batch_process(prompts, delay=0.1)
   ```

3. **Avoid Blocking Calls**
   ```python
   # BAD
   response = client.chat.completions.create(...)
   
   # GOOD
   response = await client.chat.completions.create(...)
   ```

### 🔧 Standard Practices
1. **Logging**
   ```python
   import logging
   
   logger = logging.getLogger(__name__)
   
   async def logged_generation(prompt: str):
       logger.info(f"Generating for prompt: {prompt[:50]}...")
       result, error = await safe_generate(prompt)
       if error:
           logger.error(f"Generation failed: {error}")
       return result
   ```

2. **Testing**
   ```python
   import pytest
   from unittest.mock import AsyncMock
   
   @pytest.mark.asyncio
   async def test_gpt_generation():
       client.chat.completions.create = AsyncMock()
       result = await generate_text("test")
       assert result is not None
   ```

## ✅ Assignments

### Level 1: Basic Implementation
1. Set up OpenAI API access
2. Create a simple chat completion
3. Implement basic error handling

### Level 2: Advanced Features
1. Build a function-calling system
2. Create a vision API application
3. Implement rate limiting and batching

### Bonus Challenge
- Create a FastAPI service using GPT-4
- Implement streaming responses
- Add comprehensive logging and monitoring

## 🧠 Summary
- GPT-4 offers powerful language and vision capabilities
- Function calling enables tool integration
- Best practices ensure reliable applications
- Error handling and monitoring are crucial

> **Navigation**
> - [← Gemini API](23-Python-Gemini-API.md)
> - [Course Repository →](https://github.com/SmartCodar/agentic-ai-training)
