# Day 32 – Building with LLMChain and PromptTemplate

## Overview
Learn how to create flexible and reusable prompt templates and build single-step chains using LangChain. This lesson focuses on the fundamental building blocks of LangChain applications.

## 🎯 Learning Objectives
By the end of this lesson, you will:
- Master PromptTemplate creation and usage
- Build efficient LLMChains
- Handle inputs and outputs effectively
- Create reusable prompt components

## Prerequisites
Before starting this lesson, ensure you have:
- Completed the LangChain introduction
- Understanding of basic prompt engineering
- Python programming experience
- Familiarity with LLM APIs

### ⚙️ Technical Requirements
- Python 3.11+
- LangChain library
- OpenAI or Gemini API key
- Development environment setup

## 1. Understanding PromptTemplate

### 🔍 What is PromptTemplate?
PromptTemplate is a structured way to create dynamic prompts. It's like a template engine for your LLM interactions.

```python
from langchain.prompts import PromptTemplate
from typing import List

# Basic template
basic_template = PromptTemplate(
    input_variables=["topic"],
    template="Write 5 catchy blog titles about {topic}."
)

# More complex template
blog_template = PromptTemplate(
    input_variables=["topic", "tone", "audience"],
    template="""
    Create 5 blog titles about {topic}.
    Tone: {tone}
    Target Audience: {audience}
    
    Requirements:
    - Make them engaging and clickable
    - Include numbers where appropriate
    - Keep them under 60 characters
    
    Titles:
    """
)
```

### 🛠️ Template Features
1. **Variable Validation**
   ```python
   # Template with validation
   from pydantic import BaseModel, Field
   
   class BlogPromptInput(BaseModel):
       topic: str = Field(..., min_length=3)
       tone: str = Field(..., pattern="^(professional|casual|humorous)$")
       
   template = PromptTemplate(
       input_variables=["topic", "tone"],
       template="Write about {topic} in a {tone} tone.",
       validate_template=True
   )
   ```

2. **Partial Templates**
   ```python
   # Create a partial template
   professional_template = template.partial(tone="professional")
   # Now you only need to provide 'topic'
   ```

## 2. Working with LLMChain

### 🔗 Basic Chain Setup
```python
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

def create_blog_title_chain():
    """
    Create a chain for generating blog titles
    """
    # Initialize LLM
    llm = ChatOpenAI(
        temperature=0.7,
        model="gpt-3.5-turbo"
    )
    
    # Create template
    template = PromptTemplate(
        input_variables=["topic", "count"],
        template="""
        Generate {count} engaging blog titles about {topic}.
        Make them creative and clickable.
        
        Titles:
        """
    )
    
    # Create chain
    chain = LLMChain(
        llm=llm,
        prompt=template,
        verbose=True  # See the prompts being sent
    )
    
    return chain
```

### 🔄 Using the Chain
```python
# Create and use the chain
chain = create_blog_title_chain()

# Single input
result = chain.run(
    topic="Artificial Intelligence",
    count=5
)

# Multiple inputs
results = chain.generate([
    {"topic": "Python Programming", "count": 3},
    {"topic": "Data Science", "count": 3}
])
```

### 🔍 Output Processing
```python
def process_titles(chain_output: str) -> List[str]:
    """
    Process the raw chain output into a list of titles
    
    Args:
        chain_output: Raw output from the chain
        
    Returns:
        List of cleaned titles
    """
    # Split into lines
    lines = chain_output.strip().split("\n")
    
    # Clean up titles
    titles = [
        line.strip().lstrip("0123456789-.*) ")
        for line in lines
        if line.strip()
    ]
    
    return titles

# Use the processor
raw_output = chain.run(topic="Machine Learning", count=5)
titles = process_titles(raw_output)
```

## 3. Advanced Chain Features

### 🎯 Custom Output Parsers
```python
from langchain.output_parsers import CommaSeparatedListOutputParser

# Create parser
parser = CommaSeparatedListOutputParser()

# Update template to match parser
template = PromptTemplate(
    template="Generate {count} titles about {topic}. Return them as a comma-separated list.",
    input_variables=["topic", "count"],
    output_parser=parser
)

# Create chain with parser
chain = LLMChain(llm=llm, prompt=template)
```

### 🔄 Chain Callbacks
```python
from langchain.callbacks import FileCallbackHandler
from datetime import datetime

# Create callback handler
log_file = f"chain_logs_{datetime.now():%Y%m%d}.txt"
handler = FileCallbackHandler(log_file)

# Create chain with callback
chain = LLMChain(
    llm=llm,
    prompt=template,
    callbacks=[handler]
)
```

## 4. Best Practices

### ✅ DO's
1. **Use Type Hints**
   ```python
   from typing import Dict, List
   
   def generate_titles(
       topic: str,
       count: int = 5
   ) -> List[str]:
       chain = create_blog_title_chain()
       result = chain.run(
           topic=topic,
           count=count
       )
       return process_titles(result)
   ```

2. **Implement Error Handling**
   ```python
   def safe_generate(
       topic: str,
       count: int
   ) -> tuple[List[str], str]:
       try:
           titles = generate_titles(topic, count)
           return titles, None
       except Exception as e:
           return [], f"Generation failed: {str(e)}"
   ```

3. **Add Input Validation**
   ```python
   def validate_inputs(topic: str, count: int) -> bool:
       if not topic or len(topic) < 3:
           return False
       if count < 1 or count > 10:
           return False
       return True
   ```

### ❌ DON'Ts
1. Don't hardcode model parameters
2. Don't skip error handling
3. Don't ignore input validation
4. Don't chain too many steps without testing

## ✅ Assignments

### Level 1: Basic Templates
1. Create a blog title generator
2. Add input validation
3. Process and format outputs

### Level 2: Advanced Features
1. Create a custom output parser
2. Add logging callbacks
3. Implement retry logic

### Bonus Challenge
1. Create a multi-format template
2. Add performance monitoring
3. Implement A/B testing

## 🎯 Practice Exercises

### Exercise 1: Title Generator
Create a blog title generator with different tones:
```python
def generate_titles_with_tone(
    topic: str,
    tone: str
) -> List[str]:
    # Your code here
    pass
```

### Exercise 2: Output Parser
Create a custom output parser for structured titles:
```python
class TitleParser:
    def parse(self, text: str) -> Dict[str, str]:
        # Your code here
        pass
```

## 🧠 Summary
- PromptTemplate provides structure for LLM inputs
- LLMChain simplifies single-step LLM operations
- Proper error handling and validation are crucial
- Templates can be reused and combined

## 📚 Additional Resources
1. [LangChain Templates Guide](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/)
2. [Chain Types Documentation](https://python.langchain.com/docs/modules/chains/)
3. [Best Practices Guide](https://python.langchain.com/docs/guides/best_practices)

> **Navigation**
> - [← LangChain Introduction](31-Python-LangChain-Intro.md)
> - [Sequential Chains →](33-Python-LangChain-Sequential.md)
