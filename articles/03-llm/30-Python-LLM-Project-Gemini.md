# Day 30 – Building a Content Generation System with Gemini

## Overview
In this hands-on project, you'll build a professional Content Generation API using Google's Gemini Pro and FastAPI. This project demonstrates how to adapt your LLM skills to different AI providers while maintaining clean, production-ready code.

## 🎯 Learning Objectives
By completing this project, you will:
- Master Gemini Pro API integration
- Build a multi-step content pipeline
- Compare OpenAI and Gemini approaches
- Create maintainable, scalable code
- Implement professional error handling

## Prerequisites
Before starting this project, ensure you have:
- Completed previous Gemini API lessons
- Understanding of FastAPI
- Knowledge of async programming
- Experience with API development
- Basic content generation concepts

### ⚙️ Technical Requirements
- Python 3.11+
- FastAPI and Uvicorn
- `google-generativeai` package
- Gemini API key
- VS Code or similar IDE

## 1. Project Setup

### 📁 Project Structure
Let's create our project with this structure:

```plaintext
gemini_content_api/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── gemini_service.py # Gemini integration
│   ├── schemas.py        # Data models
│   ├── config.py         # Settings
│   └── utils.py          # Helper functions
├── tests/
│   └── test_api.py       # API tests
├── .env                  # Environment variables
└── requirements.txt      # Dependencies
```

### 🔧 Initial Setup

1. **Create Virtual Environment**
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Dependencies**
```bash
pip install fastapi uvicorn google-generativeai python-dotenv pydantic
```

3. **Configure Environment**
Create `.env` file:
```plaintext
GEMINI_API_KEY=your_api_key_here
MODEL_NAME=gemini-pro
```

## 2. Core Implementation

### 📝 Data Models (schemas.py)
```python
from pydantic import BaseModel, Field
from typing import Optional

class ContentRequest(BaseModel):
    """Request model for content generation"""
    topic: str = Field(..., min_length=3, max_length=100)
    target_language: str = Field(default="Hindi")
    word_count: int = Field(default=150, ge=50, le=500)
    
    class Config:
        schema_extra = {
            "example": {
                "topic": "Future of AI in Education",
                "target_language": "Spanish",
                "word_count": 150
            }
        }

class ContentResponse(BaseModel):
    """Response model for generated content"""
    article: str
    summary: str
    translation: str
    metadata: dict
```

### 🔄 Gemini Service (gemini_service.py)
```python
import google.generativeai as genai
from typing import Optional, Dict
import os
import logging
from .schemas import ContentRequest

logger = logging.getLogger(__name__)

class ContentGenerator:
    """Handles all Gemini operations for content generation"""
    
    def __init__(self):
        """Initialize Gemini client with API key"""
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel("gemini-pro")
    
    async def generate_article(
        self,
        topic: str,
        word_count: int
    ) -> str:
        """
        Generate main article content
        
        Args:
            topic: Article topic
            word_count: Target word count
            
        Returns:
            Generated article text
        """
        prompt = f"""
        Write an informative article about {topic}.
        The article should be approximately {word_count} words.
        
        Follow this structure:
        1. Introduction: Set the context
        2. Main Points: 2-3 key ideas with examples
        3. Conclusion: Summarize and provide takeaway
        
        Make it engaging and informative.
        """
        
        try:
            response = await self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Article generation failed: {str(e)}")
            raise Exception(f"Article generation failed: {str(e)}")
    
    async def create_summary(self, text: str) -> str:
        """
        Create a 2-line summary of the article
        
        Args:
            text: Article text to summarize
            
        Returns:
            Two-line summary
        """
        prompt = f"""
        Create a clear and concise summary of this text in exactly 2 lines:
        
        {text}
        
        Make it informative and engaging.
        """
        
        try:
            response = await self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            raise Exception(f"Summary generation failed: {str(e)}")
    
    async def translate_text(
        self,
        text: str,
        target_language: str
    ) -> str:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_language: Target language
            
        Returns:
            Translated text
        """
        prompt = f"""
        Translate this text to {target_language}.
        Maintain the original tone and meaning:
        
        {text}
        """
        
        try:
            response = await self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            raise Exception(f"Translation failed: {str(e)}")
```

### 🌐 FastAPI Application (main.py)
```python
from fastapi import FastAPI, HTTPException
from .schemas import ContentRequest, ContentResponse
from .gemini_service import ContentGenerator
import logging
from time import perf_counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Content Generation API",
    description="Generate, summarize, and translate content using Gemini"
)

# Initialize generator
generator = ContentGenerator()

@app.post("/generate", response_model=ContentResponse)
async def generate_content(request: ContentRequest):
    """
    Generate article content with summary and translation
    
    Args:
        request: Content generation request
        
    Returns:
        Generated content with metadata
    """
    start_time = perf_counter()
    
    try:
        # Generate article
        logger.info(f"Generating article about: {request.topic}")
        article = await generator.generate_article(
            request.topic,
            request.word_count
        )
        
        # Create summary
        logger.info("Creating summary")
        summary = await generator.create_summary(article)
        
        # Translate summary
        logger.info(f"Translating to {request.target_language}")
        translated = await generator.translate_text(
            summary,
            request.target_language
        )
        
        duration = perf_counter() - start_time
        
        return ContentResponse(
            article=article,
            summary=summary,
            translation=translated,
            metadata={
                "topic": request.topic,
                "word_count": len(article.split()),
                "target_language": request.target_language,
                "processing_time": f"{duration:.2f}s"
            }
        )
    except Exception as e:
        logger.error(f"Content generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
```

## 3. Running the Application

### 🚀 Start the Server
```bash
uvicorn app.main:app --reload
```

### 📡 Test the API
Using curl or Postman:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Future of AI in Education",
    "target_language": "Spanish",
    "word_count": 150
  }'
```

## 4. Best Practices and Tips

### ✅ DO's
1. **Implement Retry Logic**
   ```python
   from tenacity import retry, stop_after_attempt
   
   @retry(stop=stop_after_attempt(3))
   async def generate_with_retry(prompt: str):
       try:
           response = await model.generate_content(prompt)
           return response.text
       except Exception as e:
           logger.error(f"Generation failed: {str(e)}")
           raise
   ```

2. **Add Request Validation**
   ```python
   from pydantic import validator
   
   class ContentRequest(BaseModel):
       @validator("topic")
       def validate_topic(cls, v):
           if len(v.split()) > 10:
               raise ValueError("Topic too long")
           return v.strip()
   ```

3. **Performance Monitoring**
   ```python
   async def log_performance(func):
       async def wrapper(*args, **kwargs):
           start = perf_counter()
           result = await func(*args, **kwargs)
           duration = perf_counter() - start
           
           logger.info(
               f"{func.__name__} completed",
               extra={
                   "duration": duration,
                   "success": True
               }
           )
           return result
       return wrapper
   ```

### ❌ DON'Ts
1. **Don't Skip Error Handling**
   ```python
   # BAD
   response = await model.generate_content(prompt)
   
   # GOOD
   try:
       response = await model.generate_content(prompt)
       if not response.text:
           raise ValueError("Empty response")
   except Exception as e:
       logger.error(f"Generation failed: {str(e)}")
       raise
   ```

2. **Don't Ignore Rate Limits**
   ```python
   import asyncio
   
   async def rate_limited_generate(prompts: list):
       for prompt in prompts:
           yield await model.generate_content(prompt)
           await asyncio.sleep(0.1)  # Rate limiting
   ```

### 🔧 Advanced Features
1. **Batch Processing**
   ```python
   @app.post("/generate/batch")
   async def generate_batch(requests: List[ContentRequest]):
       tasks = [
           generate_content(req)
           for req in requests
       ]
       return await asyncio.gather(*tasks)
   ```

2. **Response Caching**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=100)
   def cache_response(topic: str) -> str:
       return generate_article(topic)
   ```

## ✅ Assignments

### Level 1: Basic Implementation
1. Set up the project structure
2. Implement basic content generation
3. Add error handling

### Level 2: Advanced Features
1. Add response caching
2. Implement batch processing
3. Add performance logging

### Bonus Challenge
1. Create a comparison endpoint
2. Add content moderation
3. Implement A/B testing

## 🎯 Practice Exercises

### Exercise 1: Error Handling
Implement comprehensive error handling and logging.

### Exercise 2: Performance
Add detailed performance monitoring and optimization.

### Exercise 3: Testing
Create a test suite for the API endpoints.

## 🧠 Summary
- Built a Gemini-powered content system
- Implemented proper error handling
- Added performance monitoring
- Created a production-ready API
- Learned Gemini-specific optimizations

## 📚 Additional Resources
1. [Gemini API Documentation](https://ai.google.dev/docs)
2. [FastAPI Best Practices](https://fastapi.tiangolo.com/advanced/best-practices/)
3. [API Testing Strategies](https://www.guru99.com/api-testing.html)

> **Navigation**
> - [← LLM Project OpenAI](29-Python-LLM-Project-OpenAI.md)
> - [Course Summary →](31-Python-Course-Summary.md)
