# Day 31 – Introduction to LangChain: What, Why, and How to Begin

## Overview
Learn about LangChain, a powerful framework for building sophisticated LLM applications. This lesson introduces you to LangChain's core concepts and helps you start building more complex, context-aware AI applications.

## 🎯 Learning Objectives
By the end of this lesson, you will:
- Understand LangChain's purpose and benefits
- Master core components (LLMs, Chains, Tools, Memory)
- Set up a LangChain development environment
- Create your first LangChain application

## Prerequisites
Before starting this lesson, make sure you have:
- Strong understanding of LLM concepts
- Experience with Python async programming
- Familiarity with API integration
- Basic knowledge of prompt engineering

### ⚙️ Technical Requirements
- Python 3.11+
- LangChain package
- OpenAI or Gemini API key
- VS Code or similar IDE

## 1. Understanding LangChain

### 🔍 What is LangChain?
LangChain is a framework for building context-aware, agentic applications using LLMs. Think of it as the "Django for LLMs" - it provides a structured, modular way to build complex LLM applications.

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Simple example of a LangChain setup
llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short blog post about {topic}."
)
chain = LLMChain(llm=llm, prompt=prompt)
```

### 🌟 Key Features
1. **Modularity**
   - Reusable components
   - Easy to extend
   - Plugin architecture

2. **State Management**
   - Memory systems
   - Context preservation
   - Conversation history

3. **Tool Integration**
   - External APIs
   - Python functions
   - Database connections

## 2. Core Components

### 1️⃣ LLMs and Chat Models
```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Initialize chat model
chat = ChatOpenAI()

# Simple chat interaction
messages = [
    HumanMessage(content="What is LangChain?")
]
response = chat(messages)
```

### 2️⃣ Prompts and Templates
```python
from langchain.prompts import ChatPromptTemplate

# Create a chat template
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "Tell me about {topic}"),
])

# Format the prompt
messages = template.format_messages(topic="artificial intelligence")
```

### 3️⃣ Chains
```python
from langchain.chains import SimpleSequentialChain

# Create multiple chains
chain1 = LLMChain(llm=llm, prompt=prompt1)
chain2 = LLMChain(llm=llm, prompt=prompt2)

# Combine chains
sequential_chain = SimpleSequentialChain(
    chains=[chain1, chain2]
)
```

### 4️⃣ Memory
```python
from langchain.memory import ConversationBufferMemory

# Initialize memory
memory = ConversationBufferMemory()

# Create chain with memory
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)
```

## 3. Setting Up LangChain

### 📦 Installation
```bash
pip install langchain
pip install openai  # or google-generativeai
```

### 🔑 Configuration
```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API keys
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

## 4. Your First LangChain App

### 🔰 Basic Chain
```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def create_blog_chain():
    """
    Create a simple blog post generation chain
    """
    # Initialize LLM
    llm = OpenAI(temperature=0.7)
    
    # Create prompt template
    template = """
    Write a blog post about {topic}.
    
    The post should be:
    - Informative but concise
    - Engaging and well-structured
    - Around 200 words
    
    Blog Post:
    """
    
    prompt = PromptTemplate(
        input_variables=["topic"],
        template=template
    )
    
    # Create chain
    chain = LLMChain(
        llm=llm,
        prompt=prompt
    )
    
    return chain

# Use the chain
chain = create_blog_chain()
result = chain.run(topic="The Future of AI")
print(result)
```

### 🔄 Chain with Memory
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

def create_chat_chain():
    """
    Create a conversational chain with memory
    """
    # Initialize components
    llm = OpenAI(temperature=0.7)
    memory = ConversationBufferMemory()
    
    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    return conversation

# Use the conversation chain
conversation = create_chat_chain()
response1 = conversation.predict(input="Hi! My name is Alice.")
response2 = conversation.predict(input="What's my name?")
```

## 5. Best Practices

### ✅ DO's
1. **Use Type Hints**
   ```python
   from typing import List, Dict
   
   def process_chain_output(
       result: Dict[str, str]
   ) -> List[str]:
       return result["text"].split("\n")
   ```

2. **Handle Errors**
   ```python
   try:
       response = chain.run(input_text)
   except Exception as e:
       logger.error(f"Chain execution failed: {str(e)}")
       raise
   ```

3. **Monitor Token Usage**
   ```python
   from langchain.callbacks import get_openai_callback
   
   with get_openai_callback() as cb:
       response = chain.run(input_text)
       print(f"Total Tokens: {cb.total_tokens}")
   ```

### ❌ DON'Ts
1. Don't hardcode API keys
2. Don't ignore error handling
3. Don't skip input validation
4. Don't chain too many steps without testing

## ✅ Assignments

### Level 1: Basic Chains
1. Create a simple blog post generator
2. Add basic error handling
3. Implement input validation

### Level 2: Advanced Features
1. Create a multi-step chain
2. Add conversation memory
3. Implement token tracking

### Bonus Challenge
1. Create a Q&A system with sources
2. Add tool integration (e.g., calculator)
3. Implement streaming responses

## 🧠 Summary
- LangChain simplifies LLM application development
- Core components: LLMs, Prompts, Chains, Memory
- Easy to create complex, stateful applications
- Extensible with tools and custom components

## 📚 Additional Resources
1. [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction.html)
2. [LangChain Cookbook](https://github.com/langchain-ai/langchain/tree/master/cookbook)
3. [Community Examples](https://github.com/gkamradt/langchain-tutorials)

> **Navigation**
> - [← LLM Project Gemini](30-Python-LLM-Project-Gemini.md)
> - [Course Summary →](32-Python-Course-Summary.md)
