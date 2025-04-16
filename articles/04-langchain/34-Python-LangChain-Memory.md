# Day 34 – Memory Systems in LangChain: Building Stateful Conversations

## Overview
Learn how to implement memory systems in LangChain to create chatbots and applications that maintain context across multiple interactions. This lesson covers different memory types and their practical applications.

## 🎯 Learning Objectives
By the end of this lesson, you will:
- Understand LangChain's memory systems
- Implement different memory types
- Create context-aware conversations
- Handle conversation history effectively

## Prerequisites
Before starting this lesson, ensure you have:
- Completed Sequential Chains lesson
- Understanding of conversation flow
- Basic knowledge of state management
- Experience with async programming

### ⚙️ Technical Requirements
- Python 3.8+
- LangChain library
- OpenAI or Gemini API key
- Development environment setup

## 1. Understanding Memory Types

### 🧠 Available Memory Systems
1. **ConversationBufferMemory**
   - Stores complete conversation history
   - Simple but memory-intensive

2. **ConversationBufferWindowMemory**
   - Keeps last K interactions
   - Better for long conversations

3. **ConversationSummaryMemory**
   - Summarizes old conversations
   - Maintains context while saving tokens

4. **ConversationTokenBufferMemory**
   - Tracks token usage
   - Prevents context window overflow

```python
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationTokenBufferMemory
)
```

## 2. Implementing Basic Memory

### 🔄 Buffer Memory Example
```python
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

def create_chat_with_memory():
    """
    Create a conversation chain with basic memory
    """
    # Initialize LLM
    llm = ChatOpenAI(temperature=0.7)
    
    # Initialize memory
    memory = ConversationBufferMemory()
    
    # Create chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    return conversation

# Use the conversation
conversation = create_chat_with_memory()
response1 = conversation.predict(input="Hi! My name is Alice.")
response2 = conversation.predict(input="What's my name?")
```

### 🪟 Window Memory Example
```python
def create_windowed_chat(k: int = 3):
    """
    Create a conversation with window memory
    
    Args:
        k: Number of recent interactions to remember
    """
    memory = ConversationBufferWindowMemory(k=k)
    
    conversation = ConversationChain(
        llm=ChatOpenAI(),
        memory=memory
    )
    
    return conversation

# Test window memory
chat = create_windowed_chat(k=2)
chat.predict(input="Message 1")
chat.predict(input="Message 2")
chat.predict(input="Message 3")  # Message 1 is forgotten
```

## 3. Advanced Memory Systems

### 📝 Summary Memory
```python
from langchain.memory import ConversationSummaryMemory

def create_summary_chat():
    """
    Create a chat with summarized memory
    """
    # Initialize with summary memory
    memory = ConversationSummaryMemory(
        llm=ChatOpenAI(temperature=0)
    )
    
    conversation = ConversationChain(
        llm=ChatOpenAI(),
        memory=memory
    )
    
    return conversation

# Use summary memory
chat = create_summary_chat()
chat.predict(input="Let's talk about Python programming.")
chat.predict(input="What are some key features?")
print(chat.memory.buffer)  # View summarized history
```

### 🎯 Token Buffer Memory
```python
def create_token_chat(max_tokens: int = 2000):
    """
    Create a chat with token-aware memory
    
    Args:
        max_tokens: Maximum tokens to store
    """
    memory = ConversationTokenBufferMemory(
        llm=ChatOpenAI(),
        max_token_limit=max_tokens
    )
    
    conversation = ConversationChain(
        llm=ChatOpenAI(),
        memory=memory
    )
    
    return conversation
```

## 4. Custom Memory Implementation

### 🛠️ Creating Custom Memory
```python
from langchain.memory.chat_memory import BaseChatMemory
from typing import List, Dict, Any

class CustomMemory(BaseChatMemory):
    """
    Custom memory implementation with filtering
    """
    def __init__(self, filter_words: List[str] = None):
        super().__init__()
        self.filter_words = filter_words or []
        self.chat_history: List[Dict] = []
    
    def save_context(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, str]
    ) -> None:
        """Save context from this conversation to buffer."""
        input_str = inputs["input"]
        output_str = outputs["output"]
        
        # Filter sensitive words
        for word in self.filter_words:
            input_str = input_str.replace(word, "[FILTERED]")
            output_str = output_str.replace(word, "[FILTERED]")
        
        self.chat_history.append({
            "input": input_str,
            "output": output_str
        })
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Load memory variables from chat history."""
        return {
            "history": str(self.chat_history),
            "input": inputs["input"]
        }
```

## 5. Memory Management Best Practices

### ✅ DO's
1. **Choose Appropriate Memory Type**
   ```python
   def select_memory(conversation_type: str):
       if conversation_type == "short":
           return ConversationBufferMemory()
       elif conversation_type == "long":
           return ConversationSummaryMemory()
       else:
           return ConversationTokenBufferMemory()
   ```

2. **Implement Memory Clearing**
   ```python
   def clear_conversation(conversation: ConversationChain):
       conversation.memory.clear()
       return "Conversation history cleared"
   ```

3. **Monitor Memory Usage**
   ```python
   def check_memory_size(conversation: ConversationChain):
       if hasattr(conversation.memory, "buffer"):
           return len(conversation.memory.buffer)
       return 0
   ```

### ❌ DON'Ts
1. Don't store sensitive information
2. Don't ignore memory cleanup
3. Don't use buffer memory for long conversations
4. Don't exceed token limits

## 6. Practical Applications

### 💬 Customer Service Bot
```python
def create_service_bot():
    """
    Create a customer service bot with memory
    """
    # Use summary memory for long conversations
    memory = ConversationSummaryMemory(
        llm=ChatOpenAI(temperature=0)
    )
    
    # Create template
    template = """
    You are a helpful customer service agent.
    Previous conversation:
    {history}
    
    Human: {input}
    AI Assistant:
    """
    
    # Create chain
    conversation = ConversationChain(
        llm=ChatOpenAI(),
        memory=memory,
        prompt=PromptTemplate(
            input_variables=["history", "input"],
            template=template
        )
    )
    
    return conversation
```

### 📚 Study Tutor
```python
def create_tutor_bot():
    """
    Create a tutoring bot that remembers context
    """
    memory = ConversationBufferWindowMemory(k=5)
    
    template = """
    You are a patient tutor helping a student learn.
    Remember these key points from your conversation:
    {history}
    
    Student: {input}
    Tutor:
    """
    
    conversation = ConversationChain(
        llm=ChatOpenAI(temperature=0.7),
        memory=memory,
        prompt=PromptTemplate(
            input_variables=["history", "input"],
            template=template
        )
    )
    
    return conversation
```

## ✅ Assignments

### Level 1: Basic Memory
1. Create a simple chatbot with buffer memory
2. Implement conversation history viewing
3. Add memory clearing functionality

### Level 2: Advanced Features
1. Create a bot with summary memory
2. Add token tracking
3. Implement custom memory filtering

### Bonus Challenge
1. Create a multi-user memory system
2. Add conversation analytics
3. Implement memory persistence

## 🎯 Practice Exercises

### Exercise 1: Memory Types
Compare different memory types:
```python
def memory_comparison():
    """
    Compare different memory types
    with the same conversation
    """
    # Your code here
    pass
```

### Exercise 2: Custom Memory
Create a memory system with analytics:
```python
class AnalyticsMemory(BaseChatMemory):
    """
    Memory system that tracks conversation metrics
    """
    # Your code here
    pass
```

## 🧠 Summary
- Memory systems enable contextual conversations
- Different memory types suit different needs
- Custom memory allows specialized behavior
- Proper memory management is crucial

## 📚 Additional Resources
1. [LangChain Memory Guide](https://python.langchain.com/docs/modules/memory/)
2. [Chat Memory Types](https://python.langchain.com/docs/modules/memory/chat_messages)
3. [Memory Best Practices](https://python.langchain.com/docs/guides/memory)

> **Navigation**
> - [← Sequential Chains](33-Python-LangChain-Sequential.md)
> - [Agents →](35-Python-LangChain-Agents.md)
