# Day 20 - Role-Based Prompting and System Messages

## Overview
Role-based prompting uses structured system messages to define specific AI behaviors and personas. This approach enables consistent, context-aware responses aligned with task-specific goals such as tutoring, code review, or documentation.

## 🌟 Learning Objectives
By the end of this session, you will be able to:
- Build multi-role agents using system messages
- Define domain-specific roles like tutor, architect, reviewer
- Compose chained or staged role flows
- Build adaptive agents with switchable roles

## 📋 Prerequisites
- Day 17–19 knowledge of LLMs and prompt engineering
- Python 3.11+, OpenAI API key
- Familiarity with `pydantic`, `asyncio`, and message format

---

## 1. Role-Based Prompting Framework

### Message Structure
```python
from enum import Enum
from pydantic import BaseModel
from typing import List, Optional

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class Message(BaseModel):
    role: MessageRole
    content: str
    name: Optional[str] = None

class Conversation(BaseModel):
    messages: List[Message] = []

    def add_message(self, role: MessageRole, content: str, name=None):
        self.messages.append(Message(role=role, content=content, name=name))

    def get_context(self):
        return [msg.dict(exclude_none=True) for msg in self.messages]
```

---

## 2. Agent Framework

### Base Agent
```python
class BaseAgent:
    def __init__(self, system_prompt: str, model="gpt-3.5-turbo", temperature=0.7):
        self.model = model
        self.temperature = temperature
        self.conversation = Conversation()
        self.conversation.add_message(MessageRole.SYSTEM, system_prompt)

    async def process(self, user_input: str):
        self.conversation.add_message(MessageRole.USER, user_input)
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=self.conversation.get_context(),
                temperature=self.temperature
            )
            assistant_reply = response.choices[0].message.content
            self.conversation.add_message(MessageRole.ASSISTANT, assistant_reply)
            return assistant_reply
        except Exception as e:
            return f"Error: {e}"
```

---

## 3. Role Templates

### Sample Templates
```python
class RoleTemplate:
    PYTHON_TUTOR = """You are an expert Python tutor. Explain using simple analogies, code examples, and best practices."""
    CODE_REVIEWER = """You are a senior code reviewer. Identify bugs, suggest improvements, and explain your reasoning."""
    SYSTEM_ARCHITECT = """You are a software architect. Design scalable and secure systems with modern practices."""
```

### Role Agent Example
```python
class PythonTutor(BaseAgent):
    def __init__(self):
        super().__init__(RoleTemplate.PYTHON_TUTOR)

# Usage
agent = PythonTutor()
await agent.process("Explain list comprehension in Python")
```

---

## 4. Multi-Role Agent

### Adaptive Agent
```python
class AdaptiveAgent(BaseAgent):
    def __init__(self):
        self.roles = {
            'tutor': RoleTemplate.PYTHON_TUTOR,
            'reviewer': RoleTemplate.CODE_REVIEWER,
            'architect': RoleTemplate.SYSTEM_ARCHITECT
        }
        self.current_role = None
        super().__init__("")

    def switch_role(self, role_name):
        if role_name not in self.roles:
            raise ValueError("Invalid role")
        self.conversation = Conversation()
        self.current_role = role_name
        self.conversation.add_message(MessageRole.SYSTEM, self.roles[role_name])
```

---

## 5. Role Pipeline

### Chained Role Execution
```python
class RoleStage:
    def __init__(self, name, prompt, instructions):
        self.name = name
        self.prompt = prompt
        self.instructions = instructions

class RolePipeline:
    def __init__(self):
        self.stages = []
        self.results = {}

    def add_stage(self, stage: RoleStage):
        self.stages.append(stage)

    async def execute(self, input_text):
        for stage in self.stages:
            agent = BaseAgent(stage.prompt)
            for inst in stage.instructions:
                full_input = f"{inst}\nInput:\n{input_text}"
                result = await agent.process(full_input)
                input_text = result
                self.results[f"{stage.name}_{inst}"] = result
        return self.results
```

---

## 6. Exercises

### Basic
- Build single-role agents for:
  - SQL Tutor
  - JavaScript Debugger
  - ML Model Explainer

### Intermediate
- Create an adaptive agent that:
  - Switches between roles using a method
  - Remembers past conversations by role

### Advanced
- Implement multi-stage pipelines with:
  - Code review → Refactor → Documentation
  - Architecture → DevOps → Security Audit

---

## ✅ Summary
- Role-based prompts make AI behavior consistent and controllable
- System messages define role, tone, and constraints
- Pipelines allow staged reasoning and generation
- Adaptive agents enable dynamic persona switching

---

## 🔍 Common Issues and Fixes
| Issue | Fix |
|-------|------|
| Role confusion | Reset system prompt on switch |
| Long messages | Trim history or batch responses |
| Conflicting tone | Clarify system message boundaries |

## 📚 Additional Resources
- [OpenAI Chat API Docs](https://platform.openai.com/docs/guides/gpt)
- [LangChain Agents Guide](https://docs.langchain.com/docs/components/agents/)
- [AI Personas Framework](https://github.com/ai-personas)

## ✅ Knowledge Check
1. What is the purpose of the `system` message?
2. How do you build a multi-role agent?
3. When should you use pipelines instead of switching?
4. Why reset the conversation when switching roles?

---

> **Navigation**
> - [← Day 19: Prompt Engineering](19-Python-Prompt-Engineering.md)
> - [Day 21: Chain of Thought →](21-Python-Chain-Of-Thought.md)

