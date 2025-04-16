# Day 36 – LangChain Tools: Extending Agent Capabilities

## Overview
Learn how to create and use tools in LangChain to extend agent capabilities. This lesson covers built-in tools, custom tool creation, and integration patterns for various APIs and services.

## 🎯 Learning Objectives
By the end of this lesson, you will:
- Master LangChain's tool system
- Create custom tools
- Integrate external APIs
- Build tool-enabled applications

## Prerequisites
Before starting this lesson, ensure you have:
- Completed Agents lesson
- Understanding of API integration
- Experience with async Python
- Knowledge of error handling

### ⚙️ Technical Requirements
- Python 3.8+
- LangChain library
- OpenAI or Gemini API key
- Development environment setup

## 1. Understanding Tools

### 🔧 Tool Basics
Tools are functions that agents can use to interact with the world:
- Get information
- Perform calculations
- Call external APIs
- Execute code

```python
from langchain.tools import BaseTool
from langchain.agents import load_tools
from typing import Optional, Type
from pydantic import BaseModel

# Load built-in tools
basic_tools = load_tools(
    ["ddg-search", "calculator", "wikipedia"]
)
```

## 2. Built-in Tools

### 🔍 Search Tools
```python
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun

def setup_search_tools():
    """
    Setup search-related tools
    """
    # Web search
    search = DuckDuckGoSearchRun()
    
    # Wikipedia
    wikipedia = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper()
    )
    
    return [search, wikipedia]
```

### 🧮 Utility Tools
```python
from langchain.tools import Calculator
from langchain.tools import PythonREPLTool

def setup_utility_tools():
    """
    Setup utility tools
    """
    # Calculator
    calculator = Calculator()
    
    # Python REPL
    python_repl = PythonREPLTool()
    
    return [calculator, python_repl]
```

## 3. Creating Custom Tools

### 🛠️ Basic Custom Tool
```python
class WeatherTool(BaseTool):
    name = "weather"
    description = "Get weather information for a location"
    
    def _run(self, location: str) -> str:
        """Get weather info"""
        # Implement weather API call
        return f"Weather info for {location}"
    
    async def _arun(self, location: str) -> str:
        """Async implementation"""
        return self._run(location)
```

### 📊 Data Processing Tool
```python
class DataAnalysisTool(BaseTool):
    name = "data_analysis"
    description = "Analyze data using pandas"
    
    def _run(self, query: str) -> str:
        """
        Run data analysis
        
        Args:
            query: Analysis request
            
        Returns:
            Analysis results
        """
        try:
            import pandas as pd
            
            # Example: Load and analyze data
            df = pd.read_csv("data.csv")
            
            if "mean" in query.lower():
                return str(df.mean())
            elif "sum" in query.lower():
                return str(df.sum())
            else:
                return "Unsupported analysis type"
                
        except Exception as e:
            return f"Analysis failed: {str(e)}"
```

## 4. API Integration

### 🌐 REST API Tool
```python
import requests
from pydantic import BaseModel, Field

class APIInput(BaseModel):
    """API tool input schema"""
    endpoint: str = Field(..., description="API endpoint")
    params: dict = Field(default_factory=dict)

class RESTAPITool(BaseTool):
    name = "rest_api"
    description = "Make REST API calls"
    args_schema: Type[BaseModel] = APIInput
    
    def _run(
        self,
        endpoint: str,
        params: Optional[dict] = None
    ) -> str:
        """
        Make API request
        """
        try:
            response = requests.get(
                endpoint,
                params=params or {}
            )
            response.raise_for_status()
            return response.text
        except Exception as e:
            return f"API call failed: {str(e)}"
```

### 📝 Document Tool
```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

class DocumentTool(BaseTool):
    name = "document_processor"
    description = "Process and analyze documents"
    
    def _run(self, file_path: str) -> str:
        """
        Process document
        """
        try:
            # Load document
            loader = TextLoader(file_path)
            document = loader.load()
            
            # Split text
            splitter = CharacterTextSplitter()
            texts = splitter.split_documents(document)
            
            # Return summary
            return f"Processed {len(texts)} segments"
            
        except Exception as e:
            return f"Processing failed: {str(e)}"
```

## 5. Advanced Tool Features

### 🔒 Authentication
```python
from abc import ABC
from typing import Optional

class AuthenticatedTool(BaseTool, ABC):
    """Base class for tools requiring authentication"""
    
    api_key: str
    base_url: str
    
    def __init__(
        self,
        api_key: str,
        base_url: str
    ):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
    
    def _get_headers(self) -> dict:
        """Get authentication headers"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
```

### 🎯 Rate Limiting
```python
import time
from functools import wraps

def rate_limit(calls: int, period: float):
    """
    Rate limiting decorator
    
    Args:
        calls: Number of allowed calls
        period: Time period in seconds
    """
    def decorator(func):
        last_reset = time.time()
        calls_made = 0
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_reset, calls_made
            
            # Reset counter if period has passed
            now = time.time()
            if now - last_reset > period:
                calls_made = 0
                last_reset = now
            
            # Check rate limit
            if calls_made >= calls:
                sleep_time = period - (now - last_reset)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                calls_made = 0
                last_reset = time.time()
            
            calls_made += 1
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

class RateLimitedTool(BaseTool):
    """Tool with rate limiting"""
    
    @rate_limit(calls=60, period=60)
    def _run(self, input_str: str) -> str:
        # Implementation
        pass
```

## 6. Tool Integration Patterns

### 🔄 Tool Chaining
```python
def chain_tools(*tools):
    """
    Chain multiple tools together
    """
    def chained_run(input_str: str) -> str:
        result = input_str
        for tool in tools:
            result = tool._run(result)
        return result
    
    return chained_run

# Use chained tools
search = DuckDuckGoSearchRun()
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

chained_search = chain_tools(search, wiki)
result = chained_search("artificial intelligence")
```

### 🎯 Tool Selection
```python
def select_tool(query: str, tools: list) -> BaseTool:
    """
    Select the most appropriate tool
    
    Args:
        query: User query
        tools: Available tools
        
    Returns:
        Selected tool
    """
    # Simple keyword matching
    keywords = {
        "weather": ["weather", "temperature", "forecast"],
        "calculator": ["calculate", "math", "compute"],
        "search": ["search", "find", "look up"]
    }
    
    for tool in tools:
        if any(
            keyword in query.lower()
            for keyword in keywords.get(tool.name, [])
        ):
            return tool
    
    # Default to search
    return next(
        tool for tool in tools
        if tool.name == "search"
    )
```

## ✅ Assignments

### Level 1: Basic Tools
1. Create a simple API tool
2. Add error handling
3. Implement input validation

### Level 2: Advanced Features
1. Create an authenticated tool
2. Add rate limiting
3. Implement tool chaining

### Bonus Challenge
1. Build a file processing tool
2. Add async support
3. Implement tool metrics

## 🎯 Practice Exercises

### Exercise 1: News Tool
```python
def create_news_tool():
    """
    Create a tool that:
    1. Fetches news articles
    2. Extracts key information
    3. Returns formatted results
    """
    # Your code here
    pass
```

### Exercise 2: Data Tool
```python
def create_data_tool():
    """
    Create a tool that:
    1. Loads data from various sources
    2. Performs analysis
    3. Generates visualizations
    """
    # Your code here
    pass
```

## 🧠 Summary
- Tools extend agent capabilities
- Custom tools enable specialized tasks
- API integration patterns are crucial
- Error handling and rate limiting are important

## 📚 Additional Resources
1. [LangChain Tools Guide](https://python.langchain.com/docs/modules/agents/tools/)
2. [Custom Tools](https://python.langchain.com/docs/modules/agents/tools/custom_tools)
3. [Tool Integration](https://python.langchain.com/docs/modules/agents/tools/integration)

> **Navigation**
> - [← Agents](35-Python-LangChain-Agents.md)
> - [Document Loading →](37-Python-LangChain-Documents.md)
