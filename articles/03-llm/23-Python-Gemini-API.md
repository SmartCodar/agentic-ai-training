# Day 23 – Introduction to Gemini API

## Overview
Learn how to use Google's Gemini, a powerful multimodal LLM, through its API and SDK. Master setting up, prompting, and leveraging Gemini's unique capabilities.

## Learning Objectives
- Understand Gemini's architecture and capabilities
- Set up and configure Gemini API access
- Write effective prompts for Gemini
- Compare Gemini with other LLMs like GPT
- Implement multimodal interactions

## Prerequisites
- Strong understanding of LLM concepts
- Experience with API authentication and security
- Knowledge of prompt engineering principles
- Familiarity with Python async programming
- Understanding of multimodal data handling

### Technical Requirements
- Python 3.11+
- Google Cloud account
- `google-generativeai` package
- Gemini API key

## 1. Understanding Gemini

### Architecture and Capabilities
Gemini is Google DeepMind's multimodal LLM family:
- Processes text, code, images, audio, and video
- Supports long context (up to 1M tokens in Gemini 1.5)
- Excels at reasoning and tool usage
- Built to power Bard and Vertex AI

### Key Features
```python
# Key capabilities of Gemini
capabilities = {
    "multimodal": ["text", "code", "images", "audio", "video"],
    "context_length": "up to 1M tokens",
    "strengths": [
        "reasoning",
        "retrieval",
        "tool_usage",
        "code_generation"
    ]
}
```

## 2. Setting Up Gemini Access

### Via Google AI Studio
1. Navigate to [AI Studio](https://makersuite.google.com/app)
2. Sign in with Google account
3. Access Gemini playground

### Via Python SDK
```python
import google.generativeai as genai

# Configure API key
genai.configure(api_key='YOUR_API_KEY')

# Initialize model
model = genai.GenerativeModel('gemini-pro')
```

## 3. Basic Interactions

### Text Generation
```python
async def generate_text(prompt: str) -> str:
    response = await model.generate_content(prompt)
    return response.text

# Example usage
prompt = "Explain quantum computing in simple terms"
result = await generate_text(prompt)
```

### Structured Output
```python
async def get_structured_response(prompt: str) -> dict:
    response = await model.generate_content({
        "contents": [{
            "parts": [{
                "text": prompt + "\nRespond in JSON format."
            }]
        }]
    })
    return response.text
```

## 4. Advanced Features

### Multimodal Input
```python
from PIL import Image

async def analyze_image(image_path: str, prompt: str):
    image = Image.open(image_path)
    response = await model.generate_content([image, prompt])
    return response.text
```

### Chat Conversations
```python
async def chat_session():
    chat = model.start_chat(history=[])
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
            
        response = await chat.send_message(user_input)
        print(f"Gemini: {response.text}")
```

## 5. Best Practices and Standards

### 🎯 DO's
1. **API Key Security**
   ```python
   # Load from environment variable
   import os
   api_key = os.getenv('GEMINI_API_KEY')
   genai.configure(api_key=api_key)
   ```

2. **Structured Error Handling**
   ```python
   from typing import Optional
   
   async def safe_generate(prompt: str) -> tuple[Optional[str], Optional[str]]:
       try:
           response = await model.generate_content(prompt)
           return response.text, None
       except ValueError as ve:
           return None, f"Invalid input: {str(ve)}"
       except Exception as e:
           return None, f"Error: {str(e)}"
   ```

3. **Rate Limiting**
   ```python
   import asyncio
   from typing import List
   
   async def batch_generate(prompts: List[str], delay: float = 0.1):
       results = []
       for prompt in prompts:
           result = await safe_generate(prompt)
           results.append(result)
           await asyncio.sleep(delay)  # Respect rate limits
       return results
   ```

4. **Input Validation**
   ```python
   def validate_prompt(prompt: str) -> bool:
       if not prompt or len(prompt.strip()) == 0:
           return False
       if len(prompt) > 32000:  # Gemini's limit
           return False
       return True
   ```

### ❌ DON'Ts
1. **Never Hardcode Credentials**
   ```python
   # BAD
   api_key = "abc123..."  # Never do this
   
   # GOOD
   api_key = os.getenv('GEMINI_API_KEY')
   ```

2. **Avoid Blocking Operations**
   ```python
   # BAD
   response = model.generate_content(prompt)  # Blocking
   
   # GOOD
   response = await model.generate_content(prompt)  # Async
   ```

3. **Don't Skip Error Handling**
   ```python
   # BAD
   def quick_generate(prompt):
       return model.generate_content(prompt).text
   
   # GOOD
   async def safe_quick_generate(prompt):
       result, error = await safe_generate(prompt)
       if error:
           logger.error(f"Generation failed: {error}")
       return result
   ```

### 🔧 Standard Practices
1. **Configuration Management**
   ```python
   from pydantic import BaseSettings
   
   class GeminiConfig(BaseSettings):
       api_key: str
       model_name: str = 'gemini-pro'
       temperature: float = 0.7
       max_output_tokens: int = 2048
       
       class Config:
           env_prefix = 'GEMINI_'
   ```

2. **Logging Setup**
   ```python
   import logging
   
   logger = logging.getLogger(__name__)
   
   async def generate_with_logging(prompt: str):
       logger.info(f"Generating content for prompt length: {len(prompt)}")
       result, error = await safe_generate(prompt)
       if error:
           logger.error(f"Generation failed: {error}")
       else:
           logger.info("Generation successful")
       return result
   ```

3. **Response Processing**
   ```python
   from typing import TypedDict
   
   class GeminiResponse(TypedDict):
       content: str
       tokens_used: int
       model_used: str
       
   async def process_response(response) -> GeminiResponse:
       return {
           'content': response.text,
           'tokens_used': response.usage.total_tokens,
           'model_used': response.model
       }
   ```

4. **Testing Patterns**
   ```python
   import pytest
   
   @pytest.mark.asyncio
   async def test_gemini_generation():
       prompt = "Test prompt"
       result, error = await safe_generate(prompt)
       assert error is None
       assert isinstance(result, str)
       assert len(result) > 0
   ```

## 6. Comparing with GPT

### Key Differences
```python
comparison = {
    "context_length": {
        "gemini": "1M tokens",
        "gpt4": "128K tokens"
    },
    "multimodal": {
        "gemini": "Native support",
        "gpt4": "Vision only"
    },
    "pricing": {
        "gemini": "Generally lower",
        "gpt4": "Premium pricing"
    }
}
```

## ✅ Assignments

### Level 1: Basic Implementation
1. Set up Gemini API access
2. Create a simple text generation script
3. Implement structured output generation

### Level 2: Advanced Features
1. Build a multimodal analysis tool
2. Create a chat interface with history
3. Compare outputs with GPT-4

### Bonus Challenge
- Create a Streamlit app using Gemini
- Implement parallel processing for batch requests

## 🧠 Summary
- Gemini offers powerful multimodal capabilities
- Easy integration via SDK or API
- Strong performance in reasoning tasks
- Cost-effective alternative to GPT-4

> **Navigation**
> - [← Prompt Evaluation](22-Python-Prompt-Evaluation.md)
> - [Course Repository →](https://github.com/SmartCodar/agentic-ai-training)
