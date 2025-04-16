# Day 37 – Document Loading and Processing in LangChain

## Overview
Learn how to work with documents in LangChain, from loading different file types to processing and analyzing their content. This lesson covers document loaders, text splitting, and document transformation techniques.

## 🎯 Learning Objectives
By the end of this lesson, you will:
- Master document loading techniques
- Implement text splitting strategies
- Transform and process documents
- Create document-based applications

## Prerequisites
Before starting this lesson, ensure you have:
- Completed Tools lesson
- Understanding of file handling
- Basic text processing knowledge
- Experience with async operations

### ⚙️ Technical Requirements
- Python 3.8+
- LangChain library
- Document processing libraries (PyPDF2, docx2txt)
- Development environment setup

## 1. Document Loaders

### 📄 Basic Document Loading
```python
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader
)

def load_document(file_path: str) -> list:
    """
    Load document based on file type
    
    Args:
        file_path: Path to document
        
    Returns:
        List of document chunks
    """
    # Determine loader based on extension
    if file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    elif file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.csv'):
        loader = CSVLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    # Load document
    return loader.load()
```

### 📚 Multiple Document Types
```python
from typing import List, Dict
from pathlib import Path

class DocumentProcessor:
    """Handle multiple document types"""
    
    def __init__(self):
        self.loaders = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.csv': CSVLoader
        }
    
    def load_directory(
        self,
        dir_path: str
    ) -> Dict[str, List]:
        """
        Load all supported documents in directory
        
        Args:
            dir_path: Directory path
            
        Returns:
            Dictionary of filename: document chunks
        """
        results = {}
        directory = Path(dir_path)
        
        for file_path in directory.glob('*'):
            if file_path.suffix in self.loaders:
                loader = self.loaders[file_path.suffix]
                documents = loader(str(file_path)).load()
                results[file_path.name] = documents
        
        return results
```

## 2. Text Splitting

### ✂️ Text Splitter Types
```python
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter
)

def create_text_splitter(
    splitter_type: str = "character",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """
    Create appropriate text splitter
    
    Args:
        splitter_type: Type of splitter to use
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        Text splitter instance
    """
    if splitter_type == "character":
        return CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    elif splitter_type == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    elif splitter_type == "token":
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        raise ValueError(f"Unknown splitter type: {splitter_type}")
```

### 🔄 Document Splitting
```python
from langchain.docstore.document import Document

def split_documents(
    documents: List[Document],
    splitter_type: str = "recursive",
    chunk_size: int = 1000
) -> List[Document]:
    """
    Split documents into chunks
    
    Args:
        documents: List of documents
        splitter_type: Type of splitter
        chunk_size: Size of chunks
        
    Returns:
        List of split documents
    """
    # Create splitter
    splitter = create_text_splitter(
        splitter_type=splitter_type,
        chunk_size=chunk_size
    )
    
    # Split documents
    split_docs = splitter.split_documents(documents)
    
    return split_docs
```

## 3. Document Transformation

### 🔄 Basic Transformations
```python
from langchain.document_transformers import (
    Html2TextTransformer,
    BeautifulSoupTransformer
)

class DocumentTransformer:
    """Transform documents into different formats"""
    
    def __init__(self):
        self.html2text = Html2TextTransformer()
        self.soup = BeautifulSoupTransformer()
    
    def clean_html(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """Remove HTML formatting"""
        return self.html2text.transform_documents(documents)
    
    def extract_text(
        self,
        documents: List[Document]
    ) -> List[str]:
        """Extract plain text from documents"""
        return [doc.page_content for doc in documents]
```

### 📊 Document Analysis
```python
from collections import Counter
from typing import Dict, Any

class DocumentAnalyzer:
    """Analyze document content"""
    
    def analyze_documents(
        self,
        documents: List[Document]
    ) -> Dict[str, Any]:
        """
        Perform basic document analysis
        
        Args:
            documents: List of documents
            
        Returns:
            Analysis results
        """
        # Extract text
        texts = [doc.page_content for doc in documents]
        combined_text = " ".join(texts)
        
        # Word count
        words = combined_text.split()
        word_count = len(words)
        
        # Word frequency
        word_freq = Counter(words)
        
        return {
            "document_count": len(documents),
            "total_words": word_count,
            "unique_words": len(word_freq),
            "common_words": word_freq.most_common(10)
        }
```

## 4. Advanced Document Processing

### 🔍 Document Search
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

class DocumentSearch:
    """Search within documents"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
    
    def create_search_index(
        self,
        documents: List[Document]
    ):
        """
        Create searchable index
        """
        return Chroma.from_documents(
            documents,
            self.embeddings
        )
    
    def search_documents(
        self,
        query: str,
        db: Chroma,
        k: int = 3
    ) -> List[Document]:
        """
        Search documents
        """
        return db.similarity_search(query, k=k)
```

### 📝 Document Summarization
```python
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI

class DocumentSummarizer:
    """Summarize document content"""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
    
    def summarize_documents(
        self,
        documents: List[Document],
        chain_type: str = "map_reduce"
    ) -> str:
        """
        Create summary of documents
        
        Args:
            documents: Documents to summarize
            chain_type: Type of summarization chain
            
        Returns:
            Document summary
        """
        chain = load_summarize_chain(
            self.llm,
            chain_type=chain_type
        )
        
        return chain.run(documents)
```

## ✅ Assignments

### Level 1: Basic Processing
1. Create a document loader
2. Implement text splitting
3. Extract basic metadata

### Level 2: Advanced Features
1. Create a document search system
2. Implement summarization
3. Add document analysis

### Bonus Challenge
1. Build a document QA system
2. Add multi-format support
3. Implement caching

## 🎯 Practice Exercises

### Exercise 1: Document Processor
```python
def create_document_processor():
    """
    Create a processor that:
    1. Loads multiple formats
    2. Splits text appropriately
    3. Extracts metadata
    """
    # Your code here
    pass
```

### Exercise 2: Search System
```python
def create_search_system():
    """
    Create a system that:
    1. Indexes documents
    2. Enables semantic search
    3. Returns relevant snippets
    """
    # Your code here
    pass
```

## 🧠 Summary
- Document loaders handle various formats
- Text splitting is crucial for processing
- Transformations enable clean data
- Search and analysis enhance usability

## 📚 Additional Resources
1. [LangChain Document Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/)
2. [Text Splitting Guide](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
3. [Document Processing](https://python.langchain.com/docs/use_cases/question_answering/)

> **Navigation**
> - [← Tools](36-Python-LangChain-Tools.md)
> - [Vector Stores →](38-Python-LangChain-Vectors.md)
