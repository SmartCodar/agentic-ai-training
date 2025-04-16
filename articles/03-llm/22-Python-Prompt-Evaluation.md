# Day 22 - Prompt Evaluation and Quality Metrics

## Overview
Learn how to evaluate and improve LLM outputs using systematic evaluation frameworks and metrics.

## Learning Objectives
- Implement prompt evaluation frameworks
- Detect and prevent hallucinations
- Create automated testing systems
- Measure output quality metrics
- Design evaluation pipelines

## Prerequisites
- All previous articles (01-21)
- Understanding of LLMs (Day 17)
- Model applications (Day 18)
- Basic prompt engineering (Day 19)
- Role-based prompting (Day 20)
- Chain-of-thought prompting (Day 21)

### Technical Requirements
- Python 3.7+
- `openai` package
- `transformers` library
- `nltk` for text analysis
- `pandas` for metrics tracking

## Time Estimate
- Reading: 30 minutes
- Practice: 45 minutes
- Exercises: 45 minutes

---

## 1. Evaluation Framework

### Core Metrics System
```python
from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass
from nltk.metrics import edit_distance
import numpy as np

class MetricType(str, Enum):
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    SAFETY = "safety"

@dataclass
class EvaluationMetric:
    name: MetricType
    score: float
    details: Optional[str] = None
    confidence: float = 1.0

class PromptEvaluator:
    def __init__(self):
        self.metrics: Dict[MetricType, float] = {
            MetricType.ACCURACY: 0.0,
            MetricType.RELEVANCE: 0.0,
            MetricType.COHERENCE: 0.0,
            MetricType.SAFETY: 0.0
        }
    
    async def evaluate_response(
        self,
        prompt: str,
        response: str,
        expected: Optional[str] = None
    ) -> Dict[str, EvaluationMetric]:
        results = {}
        
        # Evaluate accuracy
        results[MetricType.ACCURACY] = await self._evaluate_accuracy(
            response, expected
        )
        
        # Evaluate relevance
        results[MetricType.RELEVANCE] = await self._evaluate_relevance(
            prompt, response
        )
        
        # Evaluate coherence
        results[MetricType.COHERENCE] = await self._evaluate_coherence(
            response
        )
        
        # Evaluate safety
        results[MetricType.SAFETY] = await self._evaluate_safety(
            response
        )
        
        return results
    
    async def _evaluate_accuracy(
        self,
        response: str,
        expected: Optional[str]
    ) -> EvaluationMetric:
        if expected:
            # Compare with expected response
            similarity = 1 - (edit_distance(response, expected) / max(len(response), len(expected)))
            return EvaluationMetric(
                name=MetricType.ACCURACY,
                score=similarity,
                details="Based on string similarity with expected response"
            )
        else:
            # Use fact-checking or other validation
            return EvaluationMetric(
                name=MetricType.ACCURACY,
                score=0.0,
                details="No expected response provided"
            )
    
    async def _evaluate_relevance(
        self,
        prompt: str,
        response: str
    ) -> EvaluationMetric:
        # Implement relevance checking
        # Example: Use embedding similarity
        return EvaluationMetric(
            name=MetricType.RELEVANCE,
            score=0.0,
            details="Relevance evaluation not implemented"
        )
    
    async def _evaluate_coherence(
        self,
        response: str
    ) -> EvaluationMetric:
        # Implement coherence checking
        # Example: Use sentence structure analysis
        return EvaluationMetric(
            name=MetricType.COHERENCE,
            score=0.0,
            details="Coherence evaluation not implemented"
        )
    
    async def _evaluate_safety(
        self,
        response: str
    ) -> EvaluationMetric:
        # Implement safety checking
        # Example: Use content moderation
        return EvaluationMetric(
            name=MetricType.SAFETY,
            score=0.0,
            details="Safety evaluation not implemented"
        )
```

### Automated Testing Framework
```python
from typing import List, Callable
import asyncio

class TestCase:
    def __init__(
        self,
        prompt: str,
        expected: Optional[str] = None,
        validators: Optional[List[Callable]] = None
    ):
        self.prompt = prompt
        self.expected = expected
        self.validators = validators or []
        self.results: Dict[str, EvaluationMetric] = {}

class PromptTestSuite:
    def __init__(self):
        self.test_cases: List[TestCase] = []
        self.evaluator = PromptEvaluator()
    
    def add_test(self, test_case: TestCase):
        self.test_cases.append(test_case)
    
    async def run_tests(self) -> Dict[str, List[EvaluationMetric]]:
        results = {}
        
        for i, test in enumerate(self.test_cases):
            # Get model response
            response = await self._get_model_response(test.prompt)
            
            # Evaluate response
            metrics = await self.evaluator.evaluate_response(
                test.prompt,
                response,
                test.expected
            )
            
            # Run custom validators
            for validator in test.validators:
                try:
                    validator(response)
                except Exception as e:
                    metrics[f"custom_validation_{i}"] = EvaluationMetric(
                        name=f"validator_{i}",
                        score=0.0,
                        details=str(e)
                    )
            
            results[f"test_{i}"] = metrics
        
        return results
    
    async def _get_model_response(self, prompt: str) -> str:
        # Implement model call
        pass
```

## 2. Evaluation Strategies

### Content Validation
```python
class ContentValidator:
    @staticmethod
    def check_factual_accuracy(text: str) -> bool:
        # Implement fact checking
        pass
    
    @staticmethod
    def check_consistency(text: str) -> bool:
        # Check for internal consistency
        pass
    
    @staticmethod
    def check_completeness(text: str) -> bool:
        # Verify all parts are present
        pass

class StyleValidator:
    @staticmethod
    def check_grammar(text: str) -> bool:
        # Check grammar
        pass
    
    @staticmethod
    def check_tone(text: str) -> bool:
        # Verify tone consistency
        pass
    
    @staticmethod
    def check_format(text: str) -> bool:
        # Validate formatting
        pass
```

### Hallucination Detection
```python
class HallucinationDetector:
    def __init__(self):
        self.knowledge_base = set()  # Add known facts
    
    def detect_hallucinations(self, text: str) -> List[str]:
        suspicious_claims = []
        # Implement detection logic
        return suspicious_claims
    
    def verify_claim(self, claim: str) -> bool:
        # Implement claim verification
        pass
    
    def get_confidence_score(self, claim: str) -> float:
        # Calculate confidence
        pass
```

## 3. Best Practices

### DO's:
1. **Use Multiple Metrics**
```python
def evaluate_comprehensively(response: str) -> Dict[str, float]:
    return {
        'accuracy': evaluate_accuracy(response),
        'relevance': evaluate_relevance(response),
        'coherence': evaluate_coherence(response),
        'safety': evaluate_safety(response)
    }
```

2. **Implement Automated Tests**
```python
def create_test_suite() -> PromptTestSuite:
    suite = PromptTestSuite()
    
    # Add test cases
    suite.add_test(TestCase(
        prompt="Explain Python decorators",
        validators=[
            lambda x: len(x) > 100,  # Length check
            lambda x: "function" in x.lower(),  # Content check
            lambda x: "@" in x  # Symbol check
        ]
    ))
    
    return suite
```

3. **Track Metrics Over Time**
```python
class MetricsTracker:
    def __init__(self):
        self.history = []
    
    def add_result(self, metrics: Dict[str, float]):
        self.history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
    
    def get_trends(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)
```

### DON'Ts:
1. **Don't Rely on Single Metric**
2. **Don't Skip Edge Cases**
3. **Don't Ignore Context**
4. **Don't Use Binary Pass/Fail Only**

## 4. Exercises

### Basic Level
1. Implement basic metrics:
   - Accuracy checker
   - Relevance scorer
   - Coherence evaluator

### Intermediate Level
1. Build automated test suite:
   - Test case generator
   - Results analyzer
   - Metrics dashboard

### Advanced Level
1. Create evaluation pipeline:
```python
class EvaluationPipeline:
    def __init__(self):
        self.stages = []
        self.results = {}
    
    def add_stage(self, name: str, evaluator: Callable):
        self.stages.append((name, evaluator))
    
    async def evaluate(self, response: str) -> Dict:
        for name, evaluator in self.stages:
            self.results[name] = await evaluator(response)
        return self.results
```

## Summary

### Key Takeaways
1. Use multiple evaluation metrics
2. Implement automated testing
3. Track performance over time
4. Handle edge cases

### Next Steps
1. Build evaluation frameworks
2. Create test suites
3. Implement monitoring
4. Develop validation tools

---

> **Navigation**
> - [← Chain-of-Thought Prompting](21-Python-Chain-Of-Thought.md)
> - [Course Repository →](https://github.com/SmartCodar/agentic-ai-training)

## Prerequisites
- Strong understanding of prompt engineering techniques
- Experience with different prompting strategies
- Knowledge of evaluation metrics and KPIs
- Familiarity with testing methodologies
- Python 3.7+ installed
- OpenAI API key
