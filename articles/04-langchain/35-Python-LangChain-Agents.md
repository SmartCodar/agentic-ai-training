# Day 35 – LangChain Agents: Building Autonomous AI Assistants

## Overview
Learn how to create autonomous agents in LangChain that can plan and execute tasks using tools. This lesson covers agent types, tool integration, and building task-specific agents.

## 🎯 Learning Objectives
By the end of this lesson, you will:
- Understand LangChain agents and their capabilities
- Create custom tools for agents
- Implement different agent types
- Build task-specific autonomous agents

## Prerequisites
Before starting this lesson, ensure you have:
- Completed Memory Systems lesson
- Understanding of LLM capabilities
- Experience with Python functions
- Knowledge of API integration

### ⚙️ Technical Requirements
- Python 3.11+
- LangChain library
- OpenAI or Gemini API key
- Development environment setup

## 1. Understanding Agents

### 🤖 What are Agents?
Agents are LLM-powered autonomous systems that can:
- Plan steps to complete tasks
- Use tools to gather information
- Make decisions based on context
- Execute actions to achieve goals

```python
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(temperature=0)

# Load basic tools
tools = load_tools(["ddg-search", "calculator"])

# Create agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

## 2. Working with Tools

### 🛠️ Creating Custom Tools
```python
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    """Inputs for weather tool"""
    location: str = Field(..., description="City name")

class WeatherTool(BaseTool):
    name = "weather"
    description = "Get weather information for a city"
    args_schema: Type[BaseModel] = WeatherInput
    
    def _run(self, location: str) -> str:
        """Get weather info (simplified example)"""
        # In real app, call weather API
        return f"Weather info for {location}"
    
    async def _arun(self, location: str) -> str:
        """Async version of _run"""
        return self._run(location)

# Create tool list
tools = [
    WeatherTool(),
    # Add more tools...
]
```

### 📚 Built-in Tools
```python
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper

# Search tool
search = DuckDuckGoSearchRun()

# Wikipedia tool
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Calculator tool
from langchain.tools import Calculator
calculator = Calculator()
```

## 3. Agent Types

### 1️⃣ Zero-Shot React Description
```python
def create_zero_shot_agent():
    """
    Create an agent that can use tools without examples
    """
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent

# Use the agent
agent = create_zero_shot_agent()
result = agent.run(
    "What's the weather in London and what's 15% of the temperature?"
)
```

### 2️⃣ Conversational React
```python
def create_chat_agent():
    """
    Create an agent that maintains conversation context
    """
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=ConversationBufferMemory(
            memory_key="chat_history"
        )
    )
    return agent

# Use chat agent
chat_agent = create_chat_agent()
chat_agent.run("Hi! Can you help me with some calculations?")
chat_agent.run("What's 15% of 85?")
```

## 4. Building Task-Specific Agents

### 📊 Research Assistant
```python
def create_research_agent():
    """
    Create an agent for research tasks
    """
    # Research-specific tools
    research_tools = [
        DuckDuckGoSearchRun(),
        WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
        Calculator()
    ]
    
    # Create agent with research prompt
    agent = initialize_agent(
        research_tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=5,
        early_stopping_method="generate",
        handle_parsing_errors=True
    )
    
    return agent

# Use research agent
researcher = create_research_agent()
research = researcher.run(
    "Research the impact of AI on healthcare and summarize key points"
)
```

### 💻 Code Assistant
```python
from langchain.tools import PythonREPLTool

def create_coding_agent():
    """
    Create an agent for coding tasks
    """
    # Coding-specific tools
    coding_tools = [
        PythonREPLTool(),
        DuckDuckGoSearchRun()
    ]
    
    # Create agent
    agent = initialize_agent(
        coding_tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=3
    )
    
    return agent

# Use coding agent
coder = create_coding_agent()
code_result = coder.run(
    "Create a function to calculate fibonacci numbers"
)
```

## 5. Advanced Agent Features

### 🔄 Custom Agent Logic
```python
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate

class CustomPromptTemplate(StringPromptTemplate):
    """Custom prompt template for agent"""
    
    template: str
    tools: list[Tool]
    
    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        
        for action, observation in intermediate_steps:
            thoughts += f"Action: {action}\n"
            thoughts += f"Observation: {observation}\n"
        
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        
        return self.template.format(**kwargs)
```

### 🎯 Output Parsing
```python
from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish

class CustomOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in text:
            return AgentFinish(
                return_values={"output": text.split("Final Answer:")[-1].strip()},
                log=text
            )
        
        # Parse out action and input
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, text)
        if not match:
            raise ValueError(f"Could not parse agent output: {text}")
        
        action = match.group(1).strip()
        action_input = match.group(2)
        
        return AgentAction(
            tool=action,
            tool_input=action_input.strip(" ").strip('"'),
            log=text
        )
```

## ✅ Assignments

### Level 1: Basic Agents
1. Create a simple search agent
2. Add calculator functionality
3. Implement basic error handling

### Level 2: Advanced Features
1. Create a custom tool
2. Implement a research agent
3. Add conversation memory

### Bonus Challenge
1. Build a multi-tool agent
2. Add custom output parsing
3. Implement agent monitoring

## 🎯 Practice Exercises

### Exercise 1: Weather Agent
```python
def create_weather_agent():
    """
    Create an agent that can:
    1. Get weather information
    2. Convert temperatures
    3. Make recommendations
    """
    # Your code here
    pass
```

### Exercise 2: Math Tutor
```python
def create_math_tutor():
    """
    Create an agent that can:
    1. Solve math problems
    2. Explain solutions
    3. Generate practice problems
    """
    # Your code here
    pass
```

## 🧠 Summary
- Agents combine LLMs with tools for autonomous tasks
- Different agent types suit different needs
- Custom tools extend agent capabilities
- Proper error handling is crucial

## 📚 Additional Resources
1. [LangChain Agents Guide](https://python.langchain.com/docs/modules/agents/)
2. [Tool Integration](https://python.langchain.com/docs/modules/agents/tools/)
3. [Agent Types](https://python.langchain.com/docs/modules/agents/agent_types/)

> **Navigation**
> - [← Memory Systems](34-Python-LangChain-Memory.md)
> - [Tools →](36-Python-LangChain-Tools.md)
