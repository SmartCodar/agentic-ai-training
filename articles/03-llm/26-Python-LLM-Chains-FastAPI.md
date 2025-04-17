# Day 26 – LLM Chains with FastAPI

## Overview
Learn to build production-ready LLM chains using FastAPI. Create a multi-step pipeline that processes text through summarization, polishing, and translation stages while maintaining reliability and performance.

## Learning Objectives
- Master prompt chaining architecture
- Build modular LLM pipelines
- Implement FastAPI microservices
- Handle errors and edge cases
- Optimize performance and costs

## Prerequisites
- Strong understanding of FastAPI
- Experience with LLM APIs (GPT-4/Gemini)
- Knowledge of async programming
- Familiarity with error handling
- Understanding of API design

### Technical Requirements
- Python 3.11+
- FastAPI and Uvicorn
- OpenAI/Google AI SDK
- API keys for chosen LLM
- Pydantic for validation

## 1. LLM Chain Architecture

### Chain Components
```python
from typing import TypedDict, Optional
from pydantic import BaseModel

class ChainStep(TypedDict):
    name: str
    description: str
    prompt_template: str
    
class ChainConfig(BaseModel):
    steps: list[ChainStep]
    max_retries: int = 3
    timeout_seconds: int = 30
```

### Basic Chain Setup
```python
from fastapi import FastAPI, HTTPException
from typing import List

app = FastAPI(title="LLM Chain API")

CHAIN_STEPS = [
    {
        "name": "summarize",
        "description": "Create a concise summary",
        "prompt_template": "Summarize the following text:\n{text}"
    },
    {
        "name": "polish",
        "description": "Improve language and grammar",
        "prompt_template": "Improve the writing of this text:\n{text}"
    },
    {
        "name": "translate",
        "description": "Translate to target language",
        "prompt_template": "Translate to {language}:\n{text}"
    }
]
```

## 2. Chain Implementation

### Step Processor
```python
from openai import OpenAI
import asyncio
from typing import Optional

class StepProcessor:
    def __init__(self, client: OpenAI):
        self.client = client
        
    async def process_step(
        self, 
        step: ChainStep, 
        text: str, 
        **kwargs
    ) -> tuple[Optional[str], Optional[str]]:
        try:
            prompt = step["prompt_template"].format(
                text=text,
                **kwargs
            )
            
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return response.choices[0].message.content, None
        except Exception as e:
            return None, f"Step {step['name']} failed: {str(e)}"
```

### Chain Executor
```python
class ChainExecutor:
    def __init__(self, steps: List[ChainStep]):
        self.steps = steps
        self.processor = StepProcessor(
            OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        )
    
    async def execute(self, text: str, **kwargs) -> dict:
        results = []
        current_text = text
        
        for step in self.steps:
            result, error = await self.processor.process_step(
                step, current_text, **kwargs
            )
            if error:
                raise HTTPException(
                    status_code=500,
                    detail=error
                )
            current_text = result
            results.append({
                "step": step["name"],
                "output": result
            })
        
        return {
            "final_output": current_text,
            "steps": results
        }
```

## 3. FastAPI Routes

### API Implementation
```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

class ChainRequest(BaseModel):
    text: str
    target_language: str = "English"
    
@app.post("/process")
async def process_text(request: ChainRequest):
    executor = ChainExecutor(CHAIN_STEPS)
    return await executor.execute(
        request.text,
        language=request.target_language
    )
```

## 4. Best Practices and Standards

### 🎯 DO's
1. **Implement Retry Logic**
   ```python
   async def retry_step(
       step: ChainStep,
       text: str,
       max_retries: int = 3
   ) -> str:
       for attempt in range(max_retries):
           result, error = await process_step(step, text)
           if not error:
               return result
           await asyncio.sleep(2 ** attempt)
       raise HTTPException(status_code=500, detail="Max retries exceeded")
   ```

2. **Add Request Validation**
   ```python
   class TextRequest(BaseModel):
       text: str
       
       @validator("text")
       def validate_text_length(cls, v):
           if len(v) < 10:
               raise ValueError("Text too short")
           if len(v) > 4000:
               raise ValueError("Text too long")
           return v
   ```

3. **Implement Logging**
   ```python
   import logging
   
   logger = logging.getLogger(__name__)
   
   async def logged_chain_execution(text: str):
       logger.info(f"Starting chain for text length: {len(text)}")
       try:
           result = await executor.execute(text)
           logger.info("Chain completed successfully")
           return result
       except Exception as e:
           logger.error(f"Chain failed: {str(e)}")
           raise
   ```

### ❌ DON'Ts
1. **Don't Chain Without Validation**
   ```python
   # BAD
   next_input = step_output  # No validation
   
   # GOOD
   if not validate_step_output(step_output):
       raise ValueError("Invalid step output")
   next_input = step_output
   ```

2. **Don't Ignore Errors**
   ```python
   # BAD
   try:
       result = await process_step(step)
   except Exception:
       result = ""  # Silent failure
   
   # GOOD
   try:
       result = await process_step(step)
   except Exception as e:
       logger.error(f"Step failed: {str(e)}")
       raise HTTPException(status_code=500, detail=str(e))
   ```

### 🔧 Standard Practices
1. **Cost Tracking**
   ```python
   class CostTracker:
       def __init__(self):
           self.total_tokens = 0
           self.total_cost = 0.0
           
       def add_usage(self, tokens: int):
           self.total_tokens += tokens
           self.total_cost += (tokens * 0.03 / 1000)  # GPT-4 rate
   ```

2. **Performance Monitoring**
   ```python
   from time import perf_counter
   
   async def timed_execution(chain_func):
       start = perf_counter()
       result = await chain_func()
       duration = perf_counter() - start
       logger.info(f"Chain executed in {duration:.2f}s")
       return result
   ```

## ✅ Assignments

### Level 1: Basic Implementation
1. Create a basic two-step chain
2. Add error handling and validation
3. Implement basic logging

### Level 2: Advanced Features
1. Add retry logic and timeouts
2. Implement cost tracking
3. Add performance monitoring

### Bonus Challenge
- Create a parallel processing chain
- Add caching for repeated requests
- Implement A/B testing of different models

## 🧠 Summary
- LLM chains enable complex processing
- FastAPI provides robust API framework
- Error handling is crucial
- Monitoring and optimization are important

> **Navigation**
> - [← GPT-4 Advanced](25-Python-GPT4-Advanced.md)
> - [Gemini Chains →](27-Python-LLM-Chains-Gemini.md)
