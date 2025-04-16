# Day 21 - Chain-of-Thought Prompting and Reasoning

## Overview
Learn how to implement Chain-of-Thought (CoT) prompting to improve LLM reasoning and problem-solving capabilities.

## Learning Objectives
- Master Chain-of-Thought prompting techniques
- Implement step-by-step reasoning in prompts
- Develop structured problem-solving approaches
- Reduce hallucinations and improve accuracy
- Create reusable CoT templates

## Prerequisites
- Strong understanding of prompt engineering
- Experience with role-based prompting
- Knowledge of reasoning and logic patterns
- Familiarity with multi-step problem solving
- Understanding of LLM capabilities and limitations

### Technical Requirements
- Python 3.7+
- OpenAI API key
- `openai` package
- `transformers` library
- `pydantic` for validation

## Time Estimate
- Reading: 30 minutes
- Practice: 45 minutes
- Exercises: 45 minutes

---

## 1. Chain-of-Thought Framework

### CoT Template System
```python
from typing import List, Dict, Optional
from pydantic import BaseModel
from enum import Enum

class ReasoningStep(BaseModel):
    step_number: int
    description: str
    result: Optional[str] = None
    explanation: Optional[str] = None

class ProblemType(str, Enum):
    MATH = "math"
    LOGIC = "logic"
    CODE = "code"
    ANALYSIS = "analysis"

class ChainOfThought:
    def __init__(self, problem_type: ProblemType):
        self.problem_type = problem_type
        self.steps: List[ReasoningStep] = []
        self.templates = {
            ProblemType.MATH: "Let's solve this math problem step by step:\n{steps}\nTherefore, the answer is: {answer}",
            ProblemType.LOGIC: "Let's think through this logically:\n{steps}\nThus, we can conclude: {answer}",
            ProblemType.CODE: "Let's break down this coding problem:\n{steps}\nThe solution is:\n{answer}",
            ProblemType.ANALYSIS: "Let's analyze this systematically:\n{steps}\nBased on this analysis: {answer}"
        }
    
    def add_step(self, description: str, result: Optional[str] = None, explanation: Optional[str] = None) -> None:
        step = ReasoningStep(
            step_number=len(self.steps) + 1,
            description=description,
            result=result,
            explanation=explanation
        )
        self.steps.append(step)
    
    def format_steps(self) -> str:
        formatted_steps = []
        for step in self.steps:
            step_text = f"{step.step_number}. {step.description}"
            if step.result:
                step_text += f"\n   Result: {step.result}"
            if step.explanation:
                step_text += f"\n   Explanation: {step.explanation}"
            formatted_steps.append(step_text)
        return "\n".join(formatted_steps)
    
    def generate_prompt(self, answer: str) -> str:
        return self.templates[self.problem_type].format(
            steps=self.format_steps(),
            answer=answer
        )
```

### Problem-Solving Agents

#### Math Problem Solver
```python
class MathProblemSolver:
    def __init__(self):
        self.cot = ChainOfThought(ProblemType.MATH)
        
    async def solve(self, problem: str) -> str:
        # Parse the problem
        self.cot.add_step(
            "Identify given values and unknowns",
            explanation="Extract numerical values and variables from the problem"
        )
        
        # Set up equations
        self.cot.add_step(
            "Set up the mathematical equation",
            explanation="Write the equation that represents the problem"
        )
        
        # Solve step by step
        self.cot.add_step(
            "Solve the equation",
            explanation="Apply mathematical operations to find the answer"
        )
        
        # Format the solution
        prompt = self.cot.generate_prompt("Final calculated result")
        return prompt

# Example usage
solver = MathProblemSolver()
prompt = await solver.solve("If a train travels 60 km in 1.5 hours, what is its average speed?")
```

#### Code Analysis Agent
```python
class CodeAnalyzer:
    def __init__(self):
        self.cot = ChainOfThought(ProblemType.CODE)
    
    async def analyze(self, code: str) -> str:
        # Parse the code
        self.cot.add_step(
            "Analyze code structure",
            explanation="Identify functions, classes, and dependencies"
        )
        
        # Check complexity
        self.cot.add_step(
            "Assess time and space complexity",
            explanation="Calculate Big O notation for critical operations"
        )
        
        # Identify improvements
        self.cot.add_step(
            "Suggest optimizations",
            explanation="List potential improvements for performance and readability"
        )
        
        # Generate solution
        prompt = self.cot.generate_prompt("Optimized code with explanations")
        return prompt

# Example usage
analyzer = CodeAnalyzer()
prompt = await analyzer.analyze("def fibonacci(n): return fibonacci(n-1) + fibonacci(n-2) if n > 1 else n")
```

## 2. Advanced CoT Techniques

### Self-Consistency Checking
```python
class SelfConsistentCoT:
    def __init__(self, n_attempts: int = 3):
        self.n_attempts = n_attempts
    
    async def solve_with_verification(self, problem: str) -> str:
        solutions = []
        
        for _ in range(self.n_attempts):
            # Generate solution with different approaches
            cot = ChainOfThought(ProblemType.MATH)
            solution = await self._generate_solution(problem, cot)
            solutions.append(solution)
        
        # Compare solutions for consistency
        if self._check_consistency(solutions):
            return solutions[0]
        else:
            return await self._resolve_inconsistency(solutions)
    
    def _check_consistency(self, solutions: List[str]) -> bool:
        # Implement solution comparison logic
        pass
    
    async def _resolve_inconsistency(self, solutions: List[str]) -> str:
        # Implement resolution strategy
        pass
```

### Recursive Reasoning
```python
class RecursiveCoT:
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
    
    async def solve(self, problem: str, depth: int = 0) -> str:
        if depth >= self.max_depth:
            return "Maximum recursion depth reached"
        
        cot = ChainOfThought(ProblemType.LOGIC)
        
        # Break down problem
        sub_problems = self._decompose_problem(problem)
        
        # Solve sub-problems recursively
        solutions = []
        for sub_problem in sub_problems:
            if self._is_simple_enough(sub_problem):
                solution = await self._solve_directly(sub_problem)
            else:
                solution = await self.solve(sub_problem, depth + 1)
            solutions.append(solution)
        
        # Combine solutions
        return self._combine_solutions(solutions)
```

## 3. Best Practices

### DO's:
1. **Break Down Complex Problems**
```python
def decompose_problem(problem: str) -> List[str]:
    """Split complex problems into manageable sub-problems."""
    # Implementation
    pass
```

2. **Validate Intermediate Steps**
```python
def validate_step(step: ReasoningStep) -> bool:
    """Verify each reasoning step is valid."""
    # Implementation
    pass
```

3. **Use Clear Transitions**
```python
def add_transition(cot: ChainOfThought, from_step: int, to_step: int) -> None:
    """Add clear transitions between reasoning steps."""
    # Implementation
    pass
```

### DON'Ts:
1. **Avoid Circular Reasoning**
```python
def check_circular_reasoning(steps: List[ReasoningStep]) -> bool:
    """Detect and prevent circular logic."""
    # Implementation
    pass
```

2. **Don't Skip Steps**
```python
def verify_step_sequence(steps: List[ReasoningStep]) -> bool:
    """Ensure no logical steps are missing."""
    # Implementation
    pass
```

## 4. Exercises

### Basic Level
1. Implement a math word problem solver using CoT
2. Create a logical reasoning chain for simple problems
3. Add validation for each reasoning step

### Intermediate Level
1. Build a code analysis system with CoT
2. Implement self-consistency checking
3. Create a recursive problem solver

### Advanced Level
1. Develop a complete reasoning framework:
```python
class ReasoningFramework:
    def __init__(self):
        self.strategies = {
            'direct': ChainOfThought,
            'recursive': RecursiveCoT,
            'self_consistent': SelfConsistentCoT
        }
    
    async def solve(self, problem: str, strategy: str) -> str:
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        solver = self.strategies[strategy]()
        return await solver.solve(problem)
```

## Summary

### Key Takeaways
1. Break down complex problems
2. Validate each step
3. Use clear reasoning chains
4. Implement verification

### Next Steps
1. Build a CoT template library
2. Implement advanced reasoning
3. Create specialized solvers
4. Add automated testing

---

> **Navigation**
> - [← Role-Based Prompting](20-Python-Role-Based-Prompting.md)
> - [Prompt Evaluation →](22-Python-Prompt-Evaluation.md)
