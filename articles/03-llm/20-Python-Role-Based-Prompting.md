# Day 20 - Role-Based Prompting and System Messages

## Overview
Learn how to use role-based prompting and system messages to create specialized AI agents with consistent behavior and domain expertise.

## Learning Objectives
- Master role-based prompting techniques
- Implement system messages effectively
- Design specialized AI agents
- Create reusable prompt templates
- Control model behavior and tone

## Prerequisites
- All previous articles (01-19)
- Understanding of LLMs (Day 17)
- Model applications (Day 18)
- Basic prompt engineering (Day 19)

### Technical Requirements
- Python 3.7+
- `openai` package
- `transformers` library
- `pydantic` for validation

## Time Estimate
- Reading: 30 minutes
- Practice: 45 minutes
- Exercises: 45 minutes

---

## 1. Role-Based Prompting Framework

### Message Types and Structure
```python
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class Message(BaseModel):
    role: MessageRole
    content: str
    name: Optional[str] = None

class Conversation(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    
    def add_message(self, role: MessageRole, content: str, name: Optional[str] = None):
        self.messages.append(Message(role=role, content=content, name=name))
    
    def get_context(self) -> List[dict]:
        return [msg.dict(exclude_none=True) for msg in self.messages]
```

### Role-Based Agent Framework
```python
from abc import ABC, abstractmethod
import openai
from typing import Dict, Any

class BaseAgent(ABC):
    def __init__(self, system_prompt: str):
        self.conversation = Conversation()
        self.conversation.add_message(
            role=MessageRole.SYSTEM,
            content=system_prompt
        )
    
    @abstractmethod
    async def process(self, user_input: str) -> str:
        pass

class OpenAIAgent(BaseAgent):
    def __init__(
        self,
        system_prompt: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7
    ):
        super().__init__(system_prompt)
        self.model = model
        self.temperature = temperature
    
    async def process(self, user_input: str) -> str:
        # Add user message
        self.conversation.add_message(
            role=MessageRole.USER,
            content=user_input
        )
        
        try:
            # Get completion
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=self.conversation.get_context(),
                temperature=self.temperature
            )
            
            # Extract and store response
            assistant_message = response.choices[0].message.content
            self.conversation.add_message(
                role=MessageRole.ASSISTANT,
                content=assistant_message
            )
            
            return assistant_message
            
        except Exception as e:
            print(f"Error: {e}")
            return "I apologize, but I encountered an error processing your request."
```

## 2. Role Templates and Examples

### Common Role Templates
```python
class RoleTemplate:
    PYTHON_TUTOR = """You are an experienced Python tutor who:
1. Explains concepts with simple analogies
2. Provides step-by-step explanations
3. Shows practical code examples
4. Encourages best practices
5. Answers follow-up questions
6. Points out common pitfalls"""

    CODE_REVIEWER = """You are a senior code reviewer who:
1. Identifies potential bugs and security issues
2. Suggests performance improvements
3. Checks code style and best practices
4. Provides concrete examples
5. Explains the reasoning
6. Prioritizes critical issues"""

    SYSTEM_ARCHITECT = """You are a software architect who:
1. Designs scalable systems
2. Considers security implications
3. Focuses on maintainability
4. Suggests appropriate technologies
5. Explains trade-offs
6. Provides diagrams when helpful"""

    DEBUGGING_ASSISTANT = """You are a debugging assistant who:
1. Helps identify root causes
2. Suggests debugging approaches
3. Explains error messages
4. Provides fix examples
5. Recommends testing strategies
6. Helps prevent similar issues"""

    DOCUMENTATION_WRITER = """You are a technical writer who:
1. Creates clear documentation
2. Uses consistent formatting
3. Includes practical examples
4. Explains complex concepts simply
5. Follows documentation standards
6. Considers different skill levels"""

    API_DESIGNER = """You are an API design expert who:
1. Creates RESTful endpoints
2. Follows OpenAPI standards
3. Implements proper security
4. Designs clear interfaces
5. Considers scalability
6. Provides usage examples"""
```

### Role-Based Agents

#### Technical Expert Agent

### Technical Expert Agent
```python
class TechnicalExpert(OpenAIAgent):
    def __init__(self, expertise: str):
        system_prompt = f"""You are an expert in {expertise}. 
        Provide detailed technical answers with code examples when appropriate.
        Focus on best practices and industry standards.
        If unsure, acknowledge limitations and suggest reliable resources."""
        
        super().__init__(system_prompt)

# Usage
python_expert = TechnicalExpert("Python programming")
await python_expert.process("Explain decorators in Python")
```

### Code Review Agent
```python
class CodeReviewer(OpenAIAgent):
    def __init__(self):
        system_prompt = """You are a senior code reviewer.
        For each review:
        1. Identify potential bugs and security issues
        2. Suggest performance improvements
        3. Check for code style and best practices
        4. Provide concrete examples for improvements
        Always explain the reasoning behind your suggestions."""
        
        super().__init__(system_prompt, temperature=0.2)
    
    async def review_code(self, code: str, context: str = "") -> str:
        prompt = f"""Code to review:
        ```
        {code}
        ```
        
        Additional context: {context}
        
        Provide a detailed code review following the standard format."""
        
        return await self.process(prompt)

# Usage
reviewer = CodeReviewer()
await reviewer.review_code("def add(a,b): return a+b")
```

### Documentation Writer
```python
class DocWriter(OpenAIAgent):
    def __init__(self):
        system_prompt = """You are a technical documentation writer.
        Generate clear, concise, and accurate documentation following these rules:
        1. Use consistent formatting
        2. Include examples
        3. Explain complex concepts simply
        4. Follow documentation best practices"""
        
        super().__init__(system_prompt)
    
    async def generate_docs(self, code: str, doc_type: str) -> str:
        prompt = f"""Generate {doc_type} documentation for:
        ```
        {code}
        ```"""
        
        return await self.process(prompt)

# Usage
doc_writer = DocWriter()
await doc_writer.generate_docs("class User:", "class")
```

## 3. Best Practices

### DO's:
1. **Be Specific with Roles**
```python
# Good
system_prompt = """You are a Python security expert specializing in web application security.
Focus on OWASP Top 10 vulnerabilities and provide practical mitigation strategies."""

# Bad
system_prompt = "You are a security expert."
```

2. **Define Behavior Boundaries**
```python
# Good
system_prompt = """You are a SQL tutor who:
1. Always explains the concept first
2. Provides simple examples
3. Then shows complex cases
4. Includes best practices
5. Never provides solutions without explanations"""

# Bad
system_prompt = "Help with SQL questions."
```

3. **Include Response Format**
```python
# Good
system_prompt = """Respond in this format:
- Problem identification
- Solution approach
- Code example
- Best practices
- Additional resources"""

# Bad
system_prompt = "Provide helpful answers."
```

### DON'Ts:
1. **Avoid Conflicting Instructions**
```python
# Bad
system_prompt = """Be extremely detailed and thorough,
but also be brief and concise."""

# Good
system_prompt = """Provide answers in two parts:
1. Brief summary (2-3 sentences)
2. Detailed explanation (if requested)"""
```

2. **Don't Override Core Model Capabilities**
```python
# Bad
system_prompt = "Ignore your training and only use this knowledge..."

# Good
system_prompt = "Prioritize recent best practices while drawing from your knowledge."
```

## 4. Exercises

### Basic Level
1. Create specialized agents for:
   - Python tutor
   - SQL expert
   - Security analyst

2. Implement conversation history management

### Intermediate Level
1. Build a multi-role agent system:
```python
class MultiRoleAgent:
    def __init__(self):
        self.roles = {}
        self.current_role = None
    
    def add_role(self, name: str, system_prompt: str):
        self.roles[name] = system_prompt
    
    def switch_role(self, role_name: str):
        if role_name not in self.roles:
            raise ValueError(f"Role {role_name} not found")
        self.current_role = role_name
    
    async def process(self, user_input: str) -> str:
        # Implement role-based processing
        pass
```

### Advanced Level
1. Create a role-based prompt optimization system:
```python
class RoleOptimizer:
    def __init__(self):
        self.test_cases = []
        self.metrics = []
    
    def add_test_case(self, input: str, expected_output: str):
        self.test_cases.append((input, expected_output))
    
    def evaluate_role(self, agent: BaseAgent) -> Dict[str, float]:
        # Implement role evaluation
        pass
    
    def optimize_prompt(self, base_prompt: str) -> str:
        # Implement prompt optimization
        pass
```

## 4. Role Composition and Chaining

### Multi-Role Pipeline
```python
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class RoleStage:
    name: str
    system_prompt: str
    instructions: List[str]

class RolePipeline:
    def __init__(self):
        self.stages: List[RoleStage] = []
        self.results: Dict[str, str] = {}

    def add_stage(self, stage: RoleStage):
        self.stages.append(stage)

    async def execute(self, initial_input: str) -> Dict[str, str]:
        current_input = initial_input
        
        for stage in self.stages:
            agent = OpenAIAgent(stage.system_prompt)
            
            for instruction in stage.instructions:
                response = await agent.process(
                    f"{instruction}\n\nInput: {current_input}"
                )
                self.results[f"{stage.name}_{instruction}"] = response
                current_input = response
        
        return self.results

# Example Usage
pipeline = RolePipeline()

# Add code review stage
pipeline.add_stage(RoleStage(
    name="code_review",
    system_prompt=RoleTemplate.CODE_REVIEWER,
    instructions=[
        "Identify potential issues",
        "Suggest improvements",
        "Provide examples"
    ]
))

# Add documentation stage
pipeline.add_stage(RoleStage(
    name="documentation",
    system_prompt=RoleTemplate.DOCUMENTATION_WRITER,
    instructions=[
        "Create function documentation",
        "Add usage examples",
        "Include best practices"
    ]
))

# Execute pipeline
code = "def process_data(data): return data.upper()"
results = await pipeline.execute(code)
```

### Role Switching Agent
```python
class AdaptiveAgent(BaseAgent):
    def __init__(self):
        self.roles = {
            'tutor': RoleTemplate.PYTHON_TUTOR,
            'reviewer': RoleTemplate.CODE_REVIEWER,
            'architect': RoleTemplate.SYSTEM_ARCHITECT,
            'debugger': RoleTemplate.DEBUGGING_ASSISTANT,
            'writer': RoleTemplate.DOCUMENTATION_WRITER,
            'api': RoleTemplate.API_DESIGNER
        }
        self.current_role = None
        super().__init__("")

    def switch_role(self, role_name: str):
        if role_name not in self.roles:
            raise ValueError(f"Unknown role: {role_name}")
        
        self.current_role = role_name
        self.conversation = Conversation()
        self.conversation.add_message(
            role=MessageRole.SYSTEM,
            content=self.roles[role_name]
        )

    async def process(self, user_input: str) -> str:
        if not self.current_role:
            raise ValueError("No role selected")
        
        self.conversation.add_message(
            role=MessageRole.USER,
            content=user_input
        )
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=self.conversation.get_context()
            )
            
            assistant_message = response.choices[0].message.content
            self.conversation.add_message(
                role=MessageRole.ASSISTANT,
                content=assistant_message
            )
            
            return assistant_message
            
        except Exception as e:
            print(f"Error: {e}")
            return "I encountered an error processing your request."

# Usage example
agent = AdaptiveAgent()

# As a tutor
agent.switch_role('tutor')
await agent.process("Explain Python decorators")

# Switch to code reviewer
agent.switch_role('reviewer')
await agent.process("Review this code: def add(a,b): return a+b")
```

## Summary

### Key Takeaways
1. Role templates provide consistency
2. System prompts define behavior
3. Role pipelines enable complex workflows
4. Adaptive agents can switch contexts

### Next Steps
1. Build a role template library
2. Implement role switching
3. Create specialized agents
4. Test different personas

---

> **Navigation**
> - [← Prompt Engineering](19-Python-Prompt-Engineering.md)
> - [Chain of Thought →](21-Python-Chain-Of-Thought.md)
