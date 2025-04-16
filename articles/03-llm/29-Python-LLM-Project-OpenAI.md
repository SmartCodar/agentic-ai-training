# Day 29 – Building a Content Generation System with OpenAI

## Overview
In this hands-on project, you'll build a professional-grade Content Generation API using OpenAI and FastAPI. This project brings together everything you've learned about LLMs, prompt engineering, and API development into a real-world application.

## 🎯 Learning Objectives
By completing this project, you will:
- Build a complete LLM-powered content generation system
- Implement production-ready API architecture
- Master prompt chaining and response handling
- Create structured, maintainable code
- Handle errors and edge cases professionally

## Prerequisites
Before starting this project, ensure you have:
- Completed previous lessons on OpenAI API
- Understanding of FastAPI basics
- Knowledge of async programming
- Familiarity with JSON and APIs
- Basic understanding of content generation

### ⚙️ Technical Requirements
- Python 3.7+
- FastAPI and Uvicorn
- OpenAI API key
- VS Code or similar IDE
- Postman/Thunder Client for testing

## 1. Project Setup

### 📁 Project Structure
First, let's create our project structure:

```plaintext
llm_content_api/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── llm_service.py    # OpenAI integration
│   ├── schemas.py        # Data models
│   ├── config.py         # Configuration
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
pip install fastapi uvicorn openai python-dotenv pydantic
```

3. **Configure Environment**
Create `.env` file:
```plaintext
OPENAI_API_KEY=your_api_key_here
MODEL_NAME=gpt-4  # or gpt-3.5-turbo for testing
```

## 2. Core Implementation

### 📝 Data Models (schemas.py)
```python
from pydantic import BaseModel, Field
from typing import Optional, List

class ContentRequest(BaseModel):
    """Request model for content generation"""
    topic: str = Field(..., min_length=3, max_length=100)
    target_language: str = Field(default="English")
    word_count: int = Field(default=200, ge=50, le=500)
    
    class Config:
        schema_extra = {
            "example": {
                "topic": "Benefits of Yoga",
                "target_language": "French",
                "word_count": 200
            }
        }

class ContentResponse(BaseModel):
    """Response model for generated content"""
    article: str
    summary: str
    translated_summary: str
    metadata: dict
```

### 🔄 LLM Service (llm_service.py)
```python
from openai import OpenAI
from typing import Optional, Dict
import os
from .schemas import ContentRequest

class ContentGenerator:
    """Handles all LLM operations for content generation"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("MODEL_NAME", "gpt-4")
    
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
        Focus on providing valuable insights and clear explanations.
        
        Use this structure:
        1. Introduction
        2. Main points (2-3)
        3. Conclusion
        
        Keep the tone professional but engaging.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return response.choices[0].message.content
        except Exception as e:
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
        Summarize this article in exactly 2 lines:
        
        {text}
        """
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return response.choices[0].message.content
        except Exception as e:
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
        Translate this text to {target_language}:
        
        {text}
        """
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Translation failed: {str(e)}")
```

### 🌐 FastAPI Application (main.py)
```python
from fastapi import FastAPI, HTTPException
from .schemas import ContentRequest, ContentResponse
from .llm_service import ContentGenerator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Content Generation API",
    description="Generate, summarize, and translate content using OpenAI"
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
        
        return ContentResponse(
            article=article,
            summary=summary,
            translated_summary=translated,
            metadata={
                "topic": request.topic,
                "word_count": len(article.split()),
                "target_language": request.target_language
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
    "topic": "Benefits of Yoga",
    "target_language": "French",
    "word_count": 200
  }'
```

## 4. Best Practices and Tips

### ✅ DO's
1. **Validate Inputs**
   ```python
   # Add to ContentRequest
   @validator("topic")
   def validate_topic(cls, v):
       if any(char in v for char in "!@#$%^&*"):
           raise ValueError("Topic contains invalid characters")
       return v
   ```

2. **Handle Rate Limits**
   ```python
   from asyncio import sleep
   
   async def with_retry(func, *args, max_retries=3):
       for attempt in range(max_retries):
           try:
               return await func(*args)
           except Exception as e:
               if "rate_limit" in str(e).lower():
                   await sleep(2 ** attempt)
                   continue
               raise
   ```

3. **Add Monitoring**
   ```python
   from time import perf_counter
   
   async def generate_with_metrics(request: ContentRequest):
       start = perf_counter()
       result = await generate_content(request)
       duration = perf_counter() - start
       
       logger.info(
           "Content generated",
           extra={
               "duration": duration,
               "topic": request.topic
           }
       )
       return result
   ```

### ❌ DON'Ts
1. Don't expose API keys in code
2. Don't skip input validation
3. Don't ignore error handling
4. Don't forget to implement rate limiting
5. Don't skip logging and monitoring

## ✅ Assignments

### Level 1: Basic Implementation
1. Set up the basic project structure
2. Implement article generation
3. Add basic error handling

### Level 2: Advanced Features
1. Add caching for similar requests
2. Implement retry logic
3. Add comprehensive logging

### Bonus Challenge
1. Add content moderation
2. Implement parallel processing
3. Add user authentication

## 🎯 Practice Exercises

### Exercise 1: Error Handling
Implement comprehensive error handling for the content generation pipeline.

### Exercise 2: Caching
Add a caching layer to store and retrieve similar requests.

### Exercise 3: Testing
Create a comprehensive test suite for the API.

## 🧠 Summary
- Built a complete content generation system
- Implemented proper error handling
- Added logging and monitoring
- Created a production-ready API
- Used best practices throughout

## 📚 Additional Resources
1. [FastAPI Documentation](https://fastapi.tiangolo.com/)
2. [OpenAI API Guide](https://platform.openai.com/docs/guides/gpt)
3. [API Security Best Practices](https://owasp.org/www-project-api-security/)

> **Navigation**
> - [← Prompt Guardrails](28-Python-Prompt-Guardrails.md)
> - [LLM Project Gemini →](30-Python-LLM-Project-Gemini.md)
