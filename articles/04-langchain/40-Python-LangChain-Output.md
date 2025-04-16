# Day 40 – Output Parsing and Structured Responses in LangChain

## Overview
Learn how to parse and structure LLM outputs in LangChain. This lesson covers output parsers, response schemas, and building applications with structured data handling.

## 🎯 Learning Objectives
By the end of this lesson, you will:
- Master output parsing techniques
- Create custom parsers
- Handle structured responses
- Build type-safe applications

## Prerequisites
Before starting this lesson, ensure you have:
- Completed Retrieval lesson
- Understanding of data structures
- Experience with Pydantic
- Knowledge of error handling

### ⚙️ Technical Requirements
- Python 3.8+
- LangChain library
- Pydantic library
- Development environment setup

## 1. Output Parser Basics

### 📝 Basic Parsers
```python
from langchain.output_parsers import (
    CommaSeparatedListOutputParser,
    StructuredOutputParser,
    ResponseSchema
)
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

class BasicParser:
    """Basic output parsing examples"""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
    
    def parse_list(self, text: str) -> list:
        """
        Parse comma-separated list
        """
        parser = CommaSeparatedListOutputParser()
        
        prompt = PromptTemplate(
            template="List items: {text}\n{format_instructions}",
            input_variables=["text"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            }
        )
        
        _input = prompt.format(text=text)
        output = self.llm.predict(_input)
        
        return parser.parse(output)
    
    def parse_structured(self, text: str) -> dict:
        """
        Parse structured output
        """
        response_schemas = [
            ResponseSchema(
                name="summary",
                description="Brief summary"
            ),
            ResponseSchema(
                name="keywords",
                description="Key terms"
            )
        ]
        
        parser = StructuredOutputParser.from_response_schemas(
            response_schemas
        )
        
        prompt = PromptTemplate(
            template="Analyze: {text}\n{format_instructions}",
            input_variables=["text"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            }
        )
        
        _input = prompt.format(text=text)
        output = self.llm.predict(_input)
        
        return parser.parse(output)
```

## 2. Custom Parsers

### 🔧 Custom Output Parser
```python
from langchain.schema import BaseOutputParser
from typing import TypeVar, Generic, Type
from pydantic import BaseModel, ValidationError

T = TypeVar('T', bound=BaseModel)

class CustomOutputParser(BaseOutputParser, Generic[T]):
    """Custom output parser for Pydantic models"""
    
    def __init__(self, pydantic_model: Type[T]):
        self.pydantic_model = pydantic_model
    
    def get_format_instructions(self) -> str:
        """Get formatting instructions"""
        fields = []
        for name, field in self.pydantic_model.__fields__.items():
            field_type = field.type_.__name__
            description = field.field_info.description
            fields.append(f"{name} ({field_type}): {description}")
        
        return f"""
        Provide output in JSON format with these fields:
        {chr(10).join(fields)}
        """
    
    def parse(self, text: str) -> T:
        """Parse output into Pydantic model"""
        try:
            # Clean and parse JSON
            cleaned = text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:-3]
            
            # Create Pydantic model
            return self.pydantic_model.parse_raw(cleaned)
            
        except ValidationError as e:
            raise ValueError(f"Failed to parse: {str(e)}")
        except Exception as e:
            raise ValueError(f"Unexpected error: {str(e)}")
```

### 📊 Data Models
```python
from pydantic import BaseModel, Field
from typing import List, Optional

class AnalysisResult(BaseModel):
    """Example Pydantic model for analysis"""
    
    summary: str = Field(
        description="Brief summary of content"
    )
    topics: List[str] = Field(
        description="Main topics discussed"
    )
    sentiment: str = Field(
        description="Overall sentiment"
    )
    confidence: float = Field(
        description="Confidence score (0-1)"
    )
    entities: Optional[List[str]] = Field(
        description="Named entities mentioned"
    )

class ContentAnalyzer:
    """Analyze content with structured output"""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.parser = CustomOutputParser(AnalysisResult)
    
    def analyze(self, content: str) -> AnalysisResult:
        """
        Analyze content
        
        Args:
            content: Text to analyze
            
        Returns:
            Structured analysis
        """
        prompt = PromptTemplate(
            template="""
            Analyze this content:
            {content}
            
            {format_instructions}
            """,
            input_variables=["content"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            }
        )
        
        _input = prompt.format(content=content)
        output = self.llm.predict(_input)
        
        return self.parser.parse(output)
```

## 3. Error Handling

### 🔍 Robust Parsing
```python
from typing import Union, Dict, Any

class RobustParser:
    """Parser with error handling"""
    
    def __init__(self, parser: BaseOutputParser):
        self.parser = parser
    
    def safe_parse(
        self,
        text: str,
        max_retries: int = 3
    ) -> Union[Any, Dict[str, Any]]:
        """
        Safely parse output with retries
        
        Args:
            text: Text to parse
            max_retries: Maximum retry attempts
            
        Returns:
            Parsed output or error dict
        """
        errors = []
        
        for attempt in range(max_retries):
            try:
                return self.parser.parse(text)
            except Exception as e:
                errors.append(str(e))
                
                if attempt < max_retries - 1:
                    # Try cleaning the output
                    text = self._clean_output(text)
                else:
                    return {
                        "error": "Parsing failed",
                        "attempts": attempt + 1,
                        "errors": errors
                    }
    
    def _clean_output(self, text: str) -> str:
        """Clean output for retry"""
        # Remove code blocks
        if "```" in text:
            text = text.split("```")[1]
        
        # Remove whitespace
        text = text.strip()
        
        # Fix common JSON issues
        text = text.replace("'", '"')
        text = text.replace("None", "null")
        
        return text
```

## 4. Building Applications

### 🌐 Structured API
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class AnalysisRequest(BaseModel):
    content: str

class AnalysisService:
    """Content analysis service"""
    
    def __init__(self):
        self.analyzer = ContentAnalyzer()
        self.parser = RobustParser(
            self.analyzer.parser
        )
    
    async def analyze_content(
        self,
        content: str
    ) -> dict:
        """
        Analyze content safely
        """
        try:
            result = self.analyzer.analyze(content)
            return result.dict()
        except Exception as e:
            return {
                "error": str(e),
                "status": "failed"
            }

# Initialize service
analysis_service = AnalysisService()

@app.post("/analyze")
async def analyze_content(request: AnalysisRequest):
    """Analyze content endpoint"""
    result = await analysis_service.analyze_content(
        request.content
    )
    
    if "error" in result:
        raise HTTPException(
            status_code=500,
            detail=result
        )
    
    return result
```

### 📊 Data Pipeline
```python
class StructuredPipeline:
    """Pipeline with structured data"""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.analyzer = ContentAnalyzer()
        self.parser = RobustParser(
            self.analyzer.parser
        )
    
    async def process_documents(
        self,
        documents: List[str]
    ) -> List[dict]:
        """
        Process multiple documents
        
        Args:
            documents: List of documents
            
        Returns:
            List of analysis results
        """
        results = []
        errors = []
        
        for doc in documents:
            try:
                analysis = await self.analyzer.analyze(doc)
                results.append(analysis.dict())
            except Exception as e:
                errors.append({
                    "document": doc[:100],
                    "error": str(e)
                })
        
        return {
            "results": results,
            "errors": errors,
            "success_rate": len(results) / len(documents)
        }
```

## ✅ Assignments

### Level 1: Basic Parsing
1. Create list parser
2. Implement structured parser
3. Handle basic errors

### Level 2: Advanced Features
1. Create custom parser
2. Add robust error handling
3. Build data pipeline

### Bonus Challenge
1. Add streaming support
2. Implement validation
3. Create complex schema

## 🎯 Practice Exercises

### Exercise 1: Custom Parser
```python
def create_custom_parser():
    """
    Create parser that:
    1. Handles complex data
    2. Validates output
    3. Provides clear errors
    """
    # Your code here
    pass
```

### Exercise 2: Data Pipeline
```python
def create_pipeline():
    """
    Build pipeline that:
    1. Processes documents
    2. Extracts structured data
    3. Handles errors gracefully
    """
    # Your code here
    pass
```

## 🧠 Summary
- Output parsing ensures structure
- Custom parsers add flexibility
- Error handling is crucial
- Validation improves reliability

## 📚 Additional Resources
1. [Output Parsers Guide](https://python.langchain.com/docs/modules/model_io/output_parsers/)
2. [Pydantic Integration](https://python.langchain.com/docs/modules/model_io/output_parsers/pydantic)
3. [Structured Output](https://python.langchain.com/docs/modules/model_io/output_parsers/structured)

> **Navigation**
> - [← Retrieval](39-Python-LangChain-Retrieval.md)
> - [Course Summary →](41-Python-LangChain-Summary.md)
