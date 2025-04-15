# Day 19 - Prompt Engineering: Zero-Shot and Few-Shot Learning

## Overview
Learn the fundamentals of prompt engineering and how to effectively communicate with Large Language Models through well-structured prompts.

## Learning Objectives
- Master zero-shot and few-shot prompting techniques
- Design effective prompt templates for various tasks
- Implement prompt engineering best practices
- Optimize prompts for better model performance
- Reduce hallucinations and improve output quality

## Prerequisites
- All previous articles (01-18)
- Understanding of LLM architectures from Day 17
- Familiarity with model applications from Day 18

### Technical Requirements
- Python 3.7+
- `openai` package
- `transformers` library
- Internet connection for API access

## Time Estimate
- Reading: 30 minutes
- Practice: 45 minutes
- Exercises: 45 minutes

---

## 1. Prompt Engineering Fundamentals

### What is Prompt Engineering?
```python
from typing import Dict, List
import openai

class PromptTemplate:
    def __init__(self, template: str):
        self.template = template
        self.variables = self._extract_variables()

    def _extract_variables(self) -> List[str]:
        # Extract {variable} placeholders from template
        import re
        return re.findall(r'\{(\w+)\}', self.template)

    def format(self, **kwargs) -> str:
        # Validate all variables are provided
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing variables: {missing}")
        return self.template.format(**kwargs)

# Example usage
summarization_template = PromptTemplate(
    "Summarize the following {content_type} in {style} style:\n\n{content}"
)

prompt = summarization_template.format(
    content_type="technical article",
    style="concise",
    content="LLMs are neural networks..."
)
```

### Zero-Shot Prompting
```python
class ZeroShotPrompt:
    def __init__(self, task: str, input_format: str):
        self.template = f"""
Task: {task}
Input format: {input_format}

Input: {{input}}
Output:"""

    def generate(self, input_text: str) -> str:
        return self.template.format(input=input_text)

# Example
classifier = ZeroShotPrompt(
    task="Classify the sentiment as positive, negative, or neutral",
    input_format="A sentence expressing an opinion"
)

prompt = classifier.generate("This product exceeded my expectations!")
```

### Few-Shot Prompting
```python
class FewShotPrompt:
    def __init__(self, task: str, examples: List[Dict[str, str]]):
        self.task = task
        self.examples = examples
        self.template = self._build_template()

    def _build_template(self) -> str:
        prompt = f"Task: {self.task}\n\n"
        
        # Add examples
        for i, example in enumerate(self.examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Input: {example['input']}\n"
            prompt += f"Output: {example['output']}\n\n"
        
        # Add new input placeholder
        prompt += "Input: {input}\nOutput:"
        return prompt

    def generate(self, input_text: str) -> str:
        return self.template.format(input=input_text)

# Example usage
translation_examples = [
    {
        "input": "Hello, how are you?",
        "output": "¡Hola, ¿cómo estás?"
    },
    {
        "input": "Good morning!",
        "output": "¡Buenos días!"
    }
]

translator = FewShotPrompt(
    task="Translate English to Spanish",
    examples=translation_examples
)

prompt = translator.generate("Thank you very much!")
```

## 2. Best Practices

### Prompt Structure
```python
class StructuredPrompt:
    def __init__(self):
        self.components = {
            'instruction': '',
            'context': '',
            'examples': [],
            'input': '',
            'format': '',
            'constraints': []
        }

    def add_instruction(self, instruction: str) -> None:
        self.components['instruction'] = instruction

    def add_context(self, context: str) -> None:
        self.components['context'] = context

    def add_example(self, input: str, output: str) -> None:
        self.components['examples'].append({'input': input, 'output': output})

    def add_format(self, format: str) -> None:
        self.components['format'] = format

    def add_constraint(self, constraint: str) -> None:
        self.components['constraints'].append(constraint)

    def build(self, input_text: str) -> str:
        prompt = []
        
        # Add instruction
        if self.components['instruction']:
            prompt.append(f"Instruction: {self.components['instruction']}")
        
        # Add context
        if self.components['context']:
            prompt.append(f"\nContext: {self.components['context']}")
        
        # Add examples
        if self.components['examples']:
            prompt.append("\nExamples:")
            for i, example in enumerate(self.components['examples'], 1):
                prompt.append(f"\nExample {i}:")
                prompt.append(f"Input: {example['input']}")
                prompt.append(f"Output: {example['output']}")
        
        # Add format
        if self.components['format']:
            prompt.append(f"\nOutput format: {self.components['format']}")
        
        # Add constraints
        if self.components['constraints']:
            prompt.append("\nConstraints:")
            for constraint in self.components['constraints']:
                prompt.append(f"- {constraint}")
        
        # Add input
        prompt.append(f"\nInput: {input_text}")
        prompt.append("Output:")
        
        return "\n".join(prompt)
```

## 3. Common Patterns and Anti-patterns

### DO's:
1. **Be Specific**
   ```python
   # Good
   prompt = "Translate the following English text to Spanish, maintaining formal tone:"
   
   # Bad
   prompt = "Translate this:"
   ```

2. **Use Structured Format**
   ```python
   # Good
   prompt = """
   Task: Summarize the article
   Style: Technical
   Length: 2-3 sentences
   Input: {text}
   """
   
   # Bad
   prompt = "Please summarize: {text}"
   ```

3. **Include Constraints**
   ```python
   # Good
   prompt = """
   Generate a Python function that:
   - Takes two parameters
   - Returns their sum
   - Includes type hints
   - Has docstring
   """
   
   # Bad
   prompt = "Write a function to add numbers"
   ```

### DON'Ts:
1. **Avoid Ambiguity**
   ```python
   # Bad
   prompt = "Make it better"
   
   # Good
   prompt = "Improve the code by:
   1. Adding error handling
   2. Optimizing performance
   3. Improving readability"
   ```

2. **Don't Overload**
   ```python
   # Bad
   prompt = "Translate to Spanish, summarize, analyze sentiment, and generate keywords"
   
   # Good
   prompt = "Translate the following text to Spanish, maintaining the original tone:"
   ```

## 4. Exercises

### Basic Level
1. Create zero-shot prompts for:
   - Text classification
   - Summarization
   - Code generation

2. Implement the `PromptTemplate` class with error handling

### Intermediate Level
1. Build a few-shot prompt system for:
   - Language translation
   - Sentiment analysis
   - Code review

2. Create a prompt evaluation system

### Advanced Level
1. Implement a prompt optimization pipeline:
   ```python
   class PromptOptimizer:
       def __init__(self, base_prompt: str):
           self.base_prompt = base_prompt
           self.variations = []
           self.results = {}

       def generate_variations(self, n: int = 5):
           # Generate prompt variations
           pass

       def evaluate_variation(self, prompt: str, test_cases: List[str]):
           # Test prompt against cases
           pass

       def optimize(self) -> str:
           # Return best performing prompt
           pass
   ```

2. Build a prompt version control system

## Summary

### Key Takeaways
1. Structured prompts improve reliability
2. Examples help with complex tasks
3. Clear constraints reduce hallucinations
4. Testing is crucial for prompt engineering

### Next Steps
1. Experiment with different prompt structures
2. Build a prompt template library
3. Implement automated testing
4. Create a prompt optimization workflow

---

> **Navigation**
> - [← LLM Applications](18-Python-LLM-Applications.md)
> - [Role Based Prompting →](20-Python-Role-Based-Prompting.md)
