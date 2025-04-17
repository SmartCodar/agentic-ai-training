# Day 19 - Prompt Engineering: Zero-Shot and Few-Shot Learning

## Overview
Prompt engineering is the art of crafting effective inputs to guide the behavior of Large Language Models (LLMs). This session focuses on building structured, optimized prompts using zero-shot and few-shot techniques.

## 🌟 Learning Objectives
By the end of this session, you'll be able to:
- Understand zero-shot vs. few-shot prompting
- Create structured prompt templates for varied tasks
- Implement reusable prompt generation classes
- Avoid anti-patterns that degrade LLM output
- Optimize for clarity, constraints, and consistency

## 📋 Prerequisites
- Day 17–18 knowledge of LLM internals and APIs
- Python string formatting and class design
- OpenAI/transformers packages installed

---

## 1. Prompt Engineering Fundamentals

### PromptTemplate Class
```python
class PromptTemplate:
    def __init__(self, template: str):
        self.template = template
        self.variables = self._extract_variables()

    def _extract_variables(self):
        import re
        return re.findall(r'\{(\w+)\}', self.template)

    def format(self, **kwargs):
        missing = set(self.variables) - set(kwargs)
        if missing:
            raise ValueError(f"Missing: {missing}")
        return self.template.format(**kwargs)
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

    def generate(self, input_text: str):
        return self.template.format(input=input_text)
```

### Few-Shot Prompting
```python
class FewShotPrompt:
    def __init__(self, task: str, examples: list):
        self.task = task
        self.examples = examples
        self.template = self._build_template()

    def _build_template(self):
        prompt = f"Task: {self.task}\n\n"
        for i, ex in enumerate(self.examples, 1):
            prompt += f"Example {i}:\nInput: {ex['input']}\nOutput: {ex['output']}\n\n"
        prompt += "Input: {input}\nOutput:"
        return prompt

    def generate(self, input_text: str):
        return self.template.format(input=input_text)
```

---

## 2. Best Practices for Prompts

### Structure Your Prompt
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

    def build(self, input_text: str):
        p = []
        if self.components['instruction']:
            p.append(f"Instruction: {self.components['instruction']}")
        if self.components['context']:
            p.append(f"Context: {self.components['context']}")
        for i, ex in enumerate(self.components['examples'], 1):
            p.append(f"\nExample {i}:\nInput: {ex['input']}\nOutput: {ex['output']}")
        if self.components['format']:
            p.append(f"\nFormat: {self.components['format']}")
        for c in self.components['constraints']:
            p.append(f"- {c}")
        p.append(f"\nInput: {input_text}\nOutput:")
        return "\n".join(p)
```

### What to Do ✅
- Be specific: "Summarize this for technical readers"
- Include format: "Output in JSON"
- Add constraints: "Max 2 sentences"
- Give examples: Few-shot improves accuracy

### What to Avoid ❌
- Vague prompts: "Make it better"
- Overloading tasks: "Summarize, translate, analyze"
- Conflicting tone: "Be concise and detailed"

---

## 3. Exercises

### Beginner:
- Build a `PromptTemplate` for text summarization.
- Create zero-shot prompts for:
  - Code generation
  - Language translation

### Intermediate:
- Use `FewShotPrompt` to:
  - Translate sentences
  - Classify tone/sentiment
- Build a prompt evaluation utility

### Advanced:
```python
class PromptOptimizer:
    def __init__(self, base_prompt):
        self.base_prompt = base_prompt
        self.variations = []

    def evaluate(self, prompt, test_cases):
        # Track token usage, latency, quality
        pass

    def optimize(self):
        # Return best-performing variation
        pass
```

---

## ✅ Summary
- Zero-shot works well for general tasks
- Few-shot boosts performance with patterns
- Structured prompting improves reliability
- Use constraints and formatting cues

---

## 🔍 Common Issues and Fixes
| Issue | Fix |
|-------|------|
| Hallucinations | Add constraints and formatting |
| Long responses | Add word limit or max tokens |
| Model misunderstanding | Add context or examples |

## 📚 Additional Resources
- [Prompt Engineering Guide](https://github.com/dair-ai/Prompt-Engineering-Guide)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [Awesome Prompt Tools](https://github.com/promptslab/Awesome-Prompt-Engineering)

## ✅ Knowledge Check
1. What’s the difference between zero-shot and few-shot?
2. Why are examples helpful in prompting?
3. How do constraints affect hallucination?
4. What are anti-patterns in prompting?

---

> **Navigation**
> - [← Day 18: LLM Applications](18-Python-LLM-Applications.md)
> - [Day 20: Role-Based Prompting →](20-Python-Role-Based-Prompting.md)

