# Day 27 – LLM Chains with Gemini

## Overview
Learn to build advanced LLM chains using Google's Gemini API. Create efficient, production-ready pipelines for text processing, leveraging Gemini's unique capabilities for summarization, enhancement, and translation.

## Learning Objectives
- Master Gemini-based chain architecture
- Build multi-step LLM pipelines
- Optimize for Gemini's capabilities
- Handle errors and retries effectively
- Implement cost-effective processing

## Prerequisites
- Strong understanding of Gemini API
- Experience with FastAPI development
- Knowledge of async programming
- Familiarity with error handling
- Understanding of prompt engineering

### Technical Requirements
- Python 3.7+
- FastAPI and Uvicorn
- `google-generativeai` package
- Gemini API key
- Pydantic for validation

## 1. Chain Architecture for Gemini

### Configuration Setup
```python
from pydantic import BaseSettings
from typing import List, Dict

class GeminiConfig(BaseSettings):
    api_key: str
    model_name: str = "gemini-pro"
    temperature: float = 0.3
    max_output_tokens: int = 1024
    
    class Config:
        env_prefix = 'GEMINI_'

class ChainConfig(BaseSettings):
    steps: List[Dict[str, str]]
    retry_attempts: int = 3
    timeout_seconds: int = 30
```

### Chain Steps Definition
```python
CHAIN_STEPS = [
    {
        "name": "summarize",
        "description": "Generate concise summary",
        "prompt_template": """
        Create a clear and concise summary of the following text.
        Focus on key points and main ideas.
        
        Text: {text}
        """
    },
    {
        "name": "enhance",
        "description": "Improve writing quality",
        "prompt_template": """
        Enhance the following text by:
        1. Improving clarity and flow
        2. Fixing grammar and style
        3. Making it more engaging
        
        Text: {text}
        """
    },
    {
        "name": "translate",
        "description": "Translate to target language",
        "prompt_template": """
        Translate the following text to {language}.
        Maintain the tone and style of the original.
        
        Text: {text}
        """
    }
]
```

## 2. Gemini Chain Implementation

### Chain Processor
```python
import google.generativeai as genai
from typing import Optional, Tuple

class GeminiProcessor:
    def __init__(self, config: GeminiConfig):
        self.config = config
        genai.configure(api_key=config.api_key)
        self.model = genai.GenerativeModel(config.model_name)
    
    async def process_step(
        self,
        step: Dict[str, str],
        text: str,
        **kwargs
    ) -> Tuple[Optional[str], Optional[str]]:
        try:
            prompt = step["prompt_template"].format(
                text=text,
                **kwargs
            )
            
            response = await self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_output_tokens
                }
            )
            return response.text, None
        except Exception as e:
            return None, f"Step {step['name']} failed: {str(e)}"
```

### Chain Executor
```python
from fastapi import HTTPException
import asyncio

class GeminiChainExecutor:
    def __init__(self, config: GeminiConfig, steps: List[Dict[str, str]]):
        self.config = config
        self.steps = steps
        self.processor = GeminiProcessor(config)
    
    async def execute_with_retry(
        self,
        step: Dict[str, str],
        text: str,
        **kwargs
    ) -> str:
        for attempt in range(self.config.retry_attempts):
            result, error = await self.processor.process_step(
                step, text, **kwargs
            )
            if not error:
                return result
            if attempt < self.config.retry_attempts - 1:
                await asyncio.sleep(2 ** attempt)
        raise HTTPException(
            status_code=500,
            detail=f"Failed after {self.config.retry_attempts} attempts"
        )
    
    async def execute_chain(self, text: str, **kwargs) -> Dict:
        results = []
        current_text = text
        
        for step in self.steps:
            current_text = await self.execute_with_retry(
                step, current_text, **kwargs
            )
            results.append({
                "step": step["name"],
                "output": current_text
            })
        
        return {
            "final_output": current_text,
            "steps": results
        }
```

## 3. FastAPI Integration

### API Routes
```python
from fastapi import FastAPI
from pydantic import BaseModel, validator

app = FastAPI(title="Gemini Chain API")

class ChainRequest(BaseModel):
    text: str
    target_language: str = "English"
    
    @validator("text")
    def validate_text(cls, v):
        if len(v) < 10:
            raise ValueError("Text too short")
        if len(v) > 30000:
            raise ValueError("Text too long for Gemini")
        return v

@app.post("/process")
async def process_text(request: ChainRequest):
    config = GeminiConfig()
    executor = GeminiChainExecutor(config, CHAIN_STEPS)
    return await executor.execute_chain(
        request.text,
        language=request.target_language
    )
```

## 4. Best Practices and Standards

### 🎯 DO's
1. **Validate Inputs**
   ```python
   def validate_gemini_input(text: str) -> bool:
       if not text or len(text.strip()) == 0:
           return False
       if len(text) > 30000:  # Gemini's limit
           return False
       return True
   ```

2. **Implement Timeouts**
   ```python
   async def timeout_wrapper(coro, timeout: int):
       try:
           return await asyncio.wait_for(coro, timeout)
       except asyncio.TimeoutError:
           raise HTTPException(
               status_code=504,
               detail="Operation timed out"
           )
   ```

3. **Add Logging**
   ```python
   import logging
   
   logger = logging.getLogger(__name__)
   
   async def logged_execution(chain_func):
       start_time = time.time()
       try:
           result = await chain_func()
           duration = time.time() - start_time
           logger.info(f"Chain completed in {duration:.2f}s")
           return result
       except Exception as e:
           logger.error(f"Chain failed: {str(e)}")
           raise
   ```

### ❌ DON'Ts
1. **Avoid Large Batches**
   ```python
   # BAD
   texts = [very_long_text for _ in range(100)]
   
   # GOOD
   async def process_in_chunks(texts: List[str], chunk_size: int = 10):
       for i in range(0, len(texts), chunk_size):
           chunk = texts[i:i + chunk_size]
           await process_chunk(chunk)
   ```

2. **Don't Skip Validation**
   ```python
   # BAD
   result = await process_text(any_input)
   
   # GOOD
   if not validate_gemini_input(text):
       raise ValueError("Invalid input for Gemini")
   result = await process_text(text)
   ```

### 🔧 Standard Practices
1. **Response Validation**
   ```python
   from pydantic import BaseModel
   
   class StepResponse(BaseModel):
       content: str
       step_name: str
       duration_ms: int
       
   def validate_step_output(
       output: str,
       step: Dict[str, str]
   ) -> StepResponse:
       if not output:
           raise ValueError(f"Empty output from {step['name']}")
       return StepResponse(
           content=output,
           step_name=step['name'],
           duration_ms=int(time.time() * 1000)
       )
   ```

2. **Performance Monitoring**
   ```python
   from dataclasses import dataclass
   from time import perf_counter
   
   @dataclass
   class ChainMetrics:
       total_tokens: int = 0
       total_time: float = 0.0
       steps_completed: int = 0
       
   async def monitored_execution(chain_func):
       metrics = ChainMetrics()
       start = perf_counter()
       result = await chain_func()
       metrics.total_time = perf_counter() - start
       return result, metrics
   ```

## ✅ Assignments

### Level 1: Basic Implementation
1. Create a basic Gemini chain
2. Add input validation
3. Implement error handling

### Level 2: Advanced Features
1. Add performance monitoring
2. Implement caching
3. Add parallel processing

### Bonus Challenge
- Create a hybrid chain using both Gemini and GPT-4
- Implement fallback mechanisms
- Add comprehensive metrics

## 🧠 Summary
- Gemini chains enable efficient processing
- Error handling and validation are crucial
- Performance monitoring helps optimization
- Proper architecture ensures reliability

> **Navigation**
> - [← LLM Chains FastAPI](26-Python-LLM-Chains-FastAPI.md)
> - [Prompt Guardrails →](28-Python-Prompt-Guardrails.md)
