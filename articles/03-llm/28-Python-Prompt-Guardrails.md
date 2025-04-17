# Day 28 – Building Safe and Reliable LLM Systems: Prompt Guardrails

## Overview
Learn how to create safe, reliable, and controlled LLM applications by implementing prompt guardrails. This lesson will teach you how to prevent common issues like bias, hallucination, and unsafe outputs while ensuring your LLM applications behave consistently and ethically.

## 🎯 Learning Objectives
By the end of this lesson, you will be able to:
- Understand what prompt guardrails are and why they're crucial
- Identify common LLM failure modes and how to prevent them
- Implement safety patterns in your prompts
- Create robust validation systems for LLM outputs
- Build production-ready safety mechanisms

## Prerequisites
Before starting this lesson, make sure you have:
- Completed the previous lessons on prompt engineering
- Understanding of LLM basics and their limitations
- Experience with Python error handling
- Familiarity with data validation
- Basic knowledge of security concepts

### ⚙️ Technical Requirements
- Python 3.11+
- Any LLM API (GPT-4 or Gemini)
- `pydantic` for validation
- Testing framework (pytest)

## 1. Understanding LLM Failure Modes

### 🔍 Common Issues
Let's explore the main ways LLMs can fail and how to prevent them:

```python
# Example of potential failures
failure_modes = {
    "hallucination": {
        "description": "Making up false information",
        "example": "Claiming non-existent research papers",
        "prevention": "Fact checking and source validation"
    },
    "bias": {
        "description": "Unfair preferences or discrimination",
        "example": "Gender or racial stereotypes",
        "prevention": "Bias detection and neutral language"
    },
    "unsafe_output": {
        "description": "Harmful or inappropriate content",
        "example": "Toxic language or unsafe advice",
        "prevention": "Content filtering and safety checks"
    }
}
```

### 📝 Understanding Each Mode
1. **Hallucination**
   - What: When LLMs generate false or made-up information
   - Why: Occurs due to pattern matching without true understanding
   - Impact: Can lead to misinformation and incorrect decisions

2. **Bias**
   - What: Unfair preferences in LLM outputs
   - Why: Training data may contain societal biases
   - Impact: Can perpetuate discrimination and unfairness

3. **Unsafe Outputs**
   - What: Harmful or inappropriate content
   - Why: LLMs may not inherently understand safety boundaries
   - Impact: Could cause harm or legal issues

## 2. Implementing Guardrails

### 🛡️ Basic Guardrail System
```python
from typing import Optional, Dict, List
from pydantic import BaseModel, validator

class GuardrailConfig(BaseModel):
    """Configuration for prompt guardrails"""
    
    # Safety settings
    content_filtering: bool = True
    fact_checking: bool = True
    bias_detection: bool = True
    
    # Thresholds
    max_toxicity_score: float = 0.7
    confidence_threshold: float = 0.8
    
    # Allowed topics/domains
    allowed_topics: List[str]
    restricted_topics: List[str]
    
    @validator("allowed_topics")
    def validate_topics(cls, v):
        if not v:
            raise ValueError("Must specify at least one allowed topic")
        return v

class SafetyCheck(BaseModel):
    """Results of safety checks"""
    is_safe: bool
    issues: List[str]
    confidence: float
```

### 🔒 Implementation Example
```python
class PromptGuardrails:
    def __init__(self, config: GuardrailConfig):
        self.config = config
        self.safety_patterns = self._load_safety_patterns()
    
    def _load_safety_patterns(self) -> Dict[str, List[str]]:
        """Load safety patterns and keywords"""
        return {
            "unsafe_patterns": [
                # Add your patterns here
                r"hack\s+into",
                r"illegal\s+access",
            ],
            "bias_patterns": [
                r"all\s+people\s+are",
                r"they\s+always",
            ]
        }
    
    async def check_prompt_safety(
        self,
        prompt: str
    ) -> SafetyCheck:
        """
        Check if a prompt is safe to use
        
        Args:
            prompt: The prompt to check
            
        Returns:
            SafetyCheck result with safety status
        """
        issues = []
        
        # Check for unsafe patterns
        for pattern in self.safety_patterns["unsafe_patterns"]:
            if re.search(pattern, prompt, re.I):
                issues.append(f"Contains unsafe pattern: {pattern}")
        
        # Check for bias
        if self.config.bias_detection:
            for pattern in self.safety_patterns["bias_patterns"]:
                if re.search(pattern, prompt, re.I):
                    issues.append(f"Potential bias detected: {pattern}")
        
        # Check allowed topics
        if not any(topic in prompt.lower() 
                  for topic in self.config.allowed_topics):
            issues.append("Topic not in allowed list")
        
        return SafetyCheck(
            is_safe=len(issues) == 0,
            issues=issues,
            confidence=0.9 if len(issues) == 0 else 0.5
        )
```

### 🔍 Output Validation
```python
class OutputValidator:
    """Validates LLM outputs for safety and quality"""
    
    def __init__(self):
        self.toxic_words = self._load_toxic_words()
        self.fact_patterns = self._load_fact_patterns()
    
    async def validate_output(
        self,
        text: str,
        context: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Validate LLM output for various quality metrics
        
        Args:
            text: The output text to validate
            context: Optional context for validation
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "is_safe": True,
            "toxicity_score": 0.0,
            "factual_accuracy": 1.0,
            "issues": []
        }
        
        # Check for toxic content
        toxic_score = await self._check_toxicity(text)
        if toxic_score > 0.7:
            results["is_safe"] = False
            results["issues"].append("High toxicity detected")
        
        # Check for factual accuracy
        if context:
            accuracy = await self._check_facts(text, context)
            results["factual_accuracy"] = accuracy
            
            if accuracy < 0.8:
                results["issues"].append("Low factual accuracy")
        
        results["toxicity_score"] = toxic_score
        return results
```

## 3. Best Practices and Guidelines

### ✅ DO's
1. **Always Validate Inputs**
   ```python
   def validate_user_input(text: str) -> bool:
       """
       Validate user input before processing
       
       Args:
           text: User input text
           
       Returns:
           True if input is valid
       """
       if not text or len(text.strip()) == 0:
           return False
           
       # Check for minimum/maximum length
       if len(text) < 10 or len(text) > 1000:
           return False
           
       # Check for basic safety
       if any(bad_word in text.lower() 
              for bad_word in UNSAFE_WORDS):
           return False
           
       return True
   ```

2. **Implement Content Filtering**
   ```python
   async def filter_content(text: str) -> str:
       """
       Filter potentially unsafe content
       
       Args:
           text: Input text
           
       Returns:
           Filtered text
       """
       # Replace unsafe words
       for word, replacement in WORD_REPLACEMENTS.items():
           text = re.sub(
               word,
               replacement,
               text,
               flags=re.IGNORECASE
           )
           
       return text
   ```

3. **Add Logging and Monitoring**
   ```python
   import logging
   from datetime import datetime
   
   class SafetyMonitor:
       def __init__(self):
           self.logger = logging.getLogger(__name__)
           
       def log_safety_event(
           self,
           event_type: str,
           details: Dict
       ):
           """
           Log safety-related events
           
           Args:
               event_type: Type of safety event
               details: Event details
           """
           self.logger.warning(
               f"Safety event: {event_type}",
               extra={
                   "timestamp": datetime.utcnow(),
                   "details": details
               }
           )
   ```

### ❌ DON'Ts
1. **Don't Skip Validation**
   ```python
   # BAD
   response = await llm.generate(user_input)
   
   # GOOD
   if not validate_user_input(user_input):
       raise ValueError("Invalid input")
   response = await llm.generate(user_input)
   ```

2. **Don't Ignore Edge Cases**
   ```python
   # BAD
   def process_output(text: str) -> str:
       return text.strip()
   
   # GOOD
   def process_output(text: str) -> str:
       if not text:
           raise ValueError("Empty output")
       
       text = text.strip()
       if len(text) < 10:
           raise ValueError("Output too short")
           
       return text
   ```

### 🔧 Production Patterns
1. **Rate Limiting**
   ```python
   from datetime import datetime, timedelta
   
   class RateLimiter:
       def __init__(self, max_requests: int, window_seconds: int):
           self.max_requests = max_requests
           self.window_seconds = window_seconds
           self.requests = []
           
       async def check_rate_limit(self, user_id: str) -> bool:
           """
           Check if user has exceeded rate limit
           
           Args:
               user_id: User identifier
               
           Returns:
               True if within rate limit
           """
           now = datetime.utcnow()
           window_start = now - timedelta(
               seconds=self.window_seconds
           )
           
           # Remove old requests
           self.requests = [
               r for r in self.requests
               if r["timestamp"] > window_start
           ]
           
           # Check current user's requests
           user_requests = len([
               r for r in self.requests
               if r["user_id"] == user_id
           ])
           
           if user_requests >= self.max_requests:
               return False
               
           self.requests.append({
               "user_id": user_id,
               "timestamp": now
           })
           return True
   ```

## ✅ Assignments

### Level 1: Basic Implementation
1. Create a simple prompt validator
   - Check for basic safety patterns
   - Implement input length validation
   - Add basic content filtering

### Level 2: Advanced Features
1. Build a comprehensive guardrail system
   - Add bias detection
   - Implement fact checking
   - Create detailed logging

### Bonus Challenge
1. Create a Testing Suite
   - Write test cases for edge cases
   - Implement automated safety checks
   - Create red team examples

## 🎯 Practice Exercises

### Exercise 1: Basic Validation
```python
# Implement this validator
def validate_prompt(prompt: str) -> bool:
    """
    Validate if a prompt is safe to use
    
    Args:
        prompt: Input prompt
        
    Returns:
        True if prompt is safe
    """
    # Your code here
    pass
```

### Exercise 2: Output Checking
```python
# Implement this checker
async def check_output_safety(
    output: str,
    context: Dict
) -> Dict[str, any]:
    """
    Check if LLM output is safe
    
    Args:
        output: LLM output
        context: Context information
        
    Returns:
        Safety check results
    """
    # Your code here
    pass
```

## 🧠 Summary
- Guardrails are essential for safe LLM systems
- Always validate inputs and outputs
- Monitor and log safety events
- Test thoroughly with edge cases
- Implement proper error handling

## 📚 Additional Resources
1. [OpenAI Safety Best Practices](https://platform.openai.com/docs/guides/safety-best-practices)
2. [Google AI Safety Standards](https://ai.google/responsibility/safety-best-practices/)
3. [LLM Security Guidelines](https://www.llmsecurity.net/)

> **Navigation**
> - [← LLM Chains Gemini](27-Python-LLM-Chains-Gemini.md)
> - [Advanced Applications →](29-Python-LLM-Advanced-Apps.md)
