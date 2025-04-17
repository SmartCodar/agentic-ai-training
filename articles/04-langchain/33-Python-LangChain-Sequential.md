# Day 33 – Sequential Chains: Building Multi-Step LLM Pipelines

## Overview
Learn how to create sophisticated multi-step LLM pipelines using Sequential Chains in LangChain. This lesson shows you how to break down complex tasks into manageable steps and process them in sequence.

## 🎯 Learning Objectives
By the end of this lesson, you will:
- Understand Sequential Chains in LangChain
- Build multi-step content processing pipelines
- Handle data flow between chain steps
- Implement error handling in pipelines

## Prerequisites
Before starting this lesson, ensure you have:
- Completed LLMChain and PromptTemplate lessons
- Understanding of basic LangChain concepts
- Experience with async Python
- Knowledge of JSON/dict handling

### ⚙️ Technical Requirements
- Python 3.11+
- LangChain library
- OpenAI or Gemini API key
- Development environment setup

## 1. Understanding Sequential Chains

### 🔗 Types of Sequential Chains
1. **SimpleSequentialChain**
   - Single input/output per step
   - Output of one step feeds into next
   
2. **SequentialChain**
   - Multiple inputs/outputs per step
   - Complex data routing

```python
from langchain.chains import SimpleSequentialChain, SequentialChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(temperature=0.7)
```

### 🔄 Simple Sequential Example
```python
# Step 1: Generate article
article_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short article about {topic}."
)
article_chain = LLMChain(llm=llm, prompt=article_prompt)

# Step 2: Create summary
summary_prompt = PromptTemplate(
    input_variables=["article"],
    template="Summarize this article in one sentence:\n\n{article}"
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# Combine chains
simple_chain = SimpleSequentialChain(
    chains=[article_chain, summary_chain],
    verbose=True
)

# Run the chain
result = simple_chain.run("AI in Education")
```

## 2. Building Complex Pipelines

### 📝 Content Processing Pipeline
```python
from typing import Dict, Any
from langchain.chains import SequentialChain

def create_content_pipeline() -> SequentialChain:
    """
    Create a multi-step content processing pipeline
    """
    # Step 1: Summarization
    summary_prompt = PromptTemplate(
        input_variables=["article"],
        template="""
        Create a concise summary of this article in 2 sentences:
        
        {article}
        """
    )
    summary_chain = LLMChain(
        llm=llm,
        prompt=summary_prompt,
        output_key="summary"
    )
    
    # Step 2: Headline Generation
    headline_prompt = PromptTemplate(
        input_variables=["summary"],
        template="""
        Create a catchy headline for an article with this summary:
        
        {summary}
        
        Make it engaging and under 60 characters.
        """
    )
    headline_chain = LLMChain(
        llm=llm,
        prompt=headline_prompt,
        output_key="headline"
    )
    
    # Step 3: Keyword Extraction
    keyword_prompt = PromptTemplate(
        input_variables=["summary", "headline"],
        template="""
        Extract 5 relevant keywords from this content:
        
        Headline: {headline}
        Summary: {summary}
        
        Format as comma-separated list.
        """
    )
    keyword_chain = LLMChain(
        llm=llm,
        prompt=keyword_prompt,
        output_key="keywords"
    )
    
    # Combine all chains
    return SequentialChain(
        chains=[summary_chain, headline_chain, keyword_chain],
        input_variables=["article"],
        output_variables=["summary", "headline", "keywords"],
        verbose=True
    )

# Use the pipeline
pipeline = create_content_pipeline()

article = """
Artificial Intelligence is revolutionizing education through personalized learning experiences.
Students now receive customized content and real-time feedback based on their learning patterns.
Institutions around the world are adopting AI-driven tools like adaptive testing, predictive analytics,
and virtual teaching assistants to increase engagement and improve outcomes.
Despite challenges like data privacy and access equity, the future of AI in classrooms looks promising.
"""

result = pipeline.run({"article": article})
print("\nProcessed Content:")
print(f"Headline: {result['headline']}")
print(f"Summary: {result['summary']}")
print(f"Keywords: {result['keywords']}")
```

## 3. Advanced Pipeline Features

### 🔍 Input/Output Validation
```python
from pydantic import BaseModel, Field

class ContentInput(BaseModel):
    article: str = Field(..., min_length=50)

class ContentOutput(BaseModel):
    summary: str
    headline: str
    keywords: str

def validate_pipeline_io(
    chain: SequentialChain,
    input_data: Dict[str, Any]
) -> ContentOutput:
    """
    Validate pipeline inputs and outputs
    """
    # Validate input
    input_model = ContentInput(article=input_data["article"])
    
    # Run chain
    result = chain.run(input_model.dict())
    
    # Validate output
    return ContentOutput(**result)
```

### 🔄 Error Handling
```python
class PipelineError(Exception):
    """Custom error for pipeline failures"""
    pass

async def safe_run_pipeline(
    chain: SequentialChain,
    input_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Safely run the pipeline with error handling
    """
    try:
        # Validate and run
        result = validate_pipeline_io(chain, input_data)
        return result.dict()
    
    except ValueError as e:
        raise PipelineError(f"Invalid input: {str(e)}")
    except Exception as e:
        raise PipelineError(f"Pipeline failed: {str(e)}")
```

### 📊 Performance Monitoring
```python
import time
from contextlib import contextmanager

@contextmanager
def pipeline_timer():
    """
    Context manager for timing pipeline execution
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        print(f"Pipeline completed in {duration:.2f} seconds")

# Use the timer
with pipeline_timer():
    result = pipeline.run({"article": article})
```

## 4. Best Practices

### ✅ DO's
1. **Break Down Complex Tasks**
   ```python
   # Instead of one complex chain
   complex_chain = LLMChain(...)
   
   # Break into smaller steps
   step1 = LLMChain(...)
   step2 = LLMChain(...)
   pipeline = SequentialChain(chains=[step1, step2])
   ```

2. **Add Logging**
   ```python
   import logging
   
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)
   
   def log_chain_step(step_name: str, input_data: dict):
       logger.info(f"Running {step_name}")
       logger.debug(f"Input: {input_data}")
   ```

3. **Implement Retries**
   ```python
   from tenacity import retry, stop_after_attempt
   
   @retry(stop=stop_after_attempt(3))
   async def retry_pipeline(chain, input_data):
       return await safe_run_pipeline(chain, input_data)
   ```

### ❌ DON'Ts
1. Don't chain too many steps
2. Don't skip error handling
3. Don't ignore performance monitoring
4. Don't hardcode model parameters

## ✅ Assignments

### Level 1: Basic Pipeline
1. Create a 3-step content pipeline
2. Add basic error handling
3. Format output as JSON

### Level 2: Advanced Features
1. Add a fourth step for tweet generation
2. Implement logging and monitoring
3. Add retry logic

### Bonus Challenge
1. Use multiple LLM providers
2. Create a FastAPI endpoint
3. Add performance benchmarking

## 🎯 Practice Exercises

### Exercise 1: News Pipeline
Create a news article processing pipeline:
```python
def create_news_pipeline():
    """
    Create a pipeline that:
    1. Summarizes news article
    2. Generates headline
    3. Extracts key points
    4. Suggests social media posts
    """
    # Your code here
    pass
```

### Exercise 2: Error Handling
Implement comprehensive error handling:
```python
class PipelineResult:
    def __init__(self, success: bool, data: dict = None, error: str = None):
        self.success = success
        self.data = data
        self.error = error

def run_safe_pipeline(chain, input_data) -> PipelineResult:
    # Your code here
    pass
```

## 🧠 Summary
- Sequential Chains enable complex LLM workflows
- Proper error handling is crucial
- Break down tasks into manageable steps
- Monitor and optimize performance

## 📚 Additional Resources
1. [LangChain Sequential Chains](https://python.langchain.com/docs/modules/chains/sequential_chains)
2. [Error Handling Best Practices](https://python.langchain.com/docs/guides/debugging)
3. [Performance Optimization](https://python.langchain.com/docs/guides/deployment)

> **Navigation**
> - [← LangChain Templates](32-Python-LangChain-Templates.md)
> - [Memory and State →](34-Python-LangChain-Memory.md)
