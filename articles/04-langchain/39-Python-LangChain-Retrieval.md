# Day 39 – Retrieval Augmented Generation (RAG) in LangChain

## Overview
Learn how to implement Retrieval Augmented Generation (RAG) in LangChain to enhance LLM responses with relevant context from your document collection. This lesson covers retrieval strategies, context augmentation, and building RAG applications.

## 🎯 Learning Objectives
By the end of this lesson, you will:
- Understand RAG architecture
- Implement retrieval strategies
- Build context-aware LLM systems
- Create RAG applications

## Prerequisites
Before starting this lesson, ensure you have:
- Completed Vector Stores lesson
- Understanding of embeddings
- Experience with LLMs
- Knowledge of document processing

### ⚙️ Technical Requirements
- Python 3.11+
- LangChain library
- Vector store (Chroma, FAISS)
- Development environment setup

## 1. RAG Fundamentals

### 🔄 Basic RAG Pipeline
```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

class RAGSystem:
    """Basic RAG implementation"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0)
    
    def create_retriever(
        self,
        documents,
        k: int = 3
    ):
        """
        Create document retriever
        
        Args:
            documents: Document collection
            k: Number of documents to retrieve
        """
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents,
            self.embeddings
        )
        
        # Create base retriever
        base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        # Add contextual compression
        compressor = LLMChainExtractor.from_llm(self.llm)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        return retriever
    
    def create_qa_chain(self, retriever):
        """Create QA chain with retriever"""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
```

## 2. Advanced Retrieval Strategies

### 🎯 Multi-Query Retrieval
```python
from langchain.retrievers import MultiQueryRetriever

class EnhancedRetriever:
    """Advanced retrieval techniques"""
    
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
    
    def create_multi_query_retriever(
        self,
        vectorstore,
        n_queries: int = 3
    ):
        """
        Create multi-query retriever
        
        Args:
            vectorstore: Vector store
            n_queries: Number of queries to generate
        """
        retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(),
            llm=self.llm,
            n_queries=n_queries
        )
        return retriever
    
    def hybrid_search(
        self,
        vectorstore,
        query: str,
        k: int = 3
    ):
        """
        Combine keyword and semantic search
        """
        # Semantic search
        semantic_docs = vectorstore.similarity_search(
            query,
            k=k
        )
        
        # Keyword search
        keyword_docs = vectorstore.keyword_search(
            query,
            k=k
        )
        
        # Combine and deduplicate
        all_docs = list({
            doc.page_content: doc
            for doc in semantic_docs + keyword_docs
        }.values())
        
        return all_docs[:k]
```

## 3. Context Processing

### 📝 Context Augmentation
```python
from typing import List
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

class ContextProcessor:
    """Process and augment context"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def summarize_context(
        self,
        documents: List[Document]
    ) -> str:
        """
        Summarize retrieved context
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Summarized context
        """
        template = """
        Summarize the following context:
        
        {context}
        
        Summary:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context"]
        )
        
        context = "\n\n".join(
            [doc.page_content for doc in documents]
        )
        
        return self.llm.predict(
            prompt.format(context=context)
        )
    
    def rank_relevance(
        self,
        query: str,
        documents: List[Document]
    ) -> List[tuple]:
        """
        Rank documents by relevance
        """
        template = """
        Rate the relevance of this document to the query:
        Query: {query}
        Document: {document}
        
        Rate from 0-10:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["query", "document"]
        )
        
        rankings = []
        for doc in documents:
            score = float(
                self.llm.predict(
                    prompt.format(
                        query=query,
                        document=doc.page_content
                    )
                )
            )
            rankings.append((doc, score))
        
        return sorted(
            rankings,
            key=lambda x: x[1],
            reverse=True
        )
```

## 4. Building RAG Applications

### 💬 Question Answering System
```python
class RAGQuestionAnswering:
    """Complete RAG-based QA system"""
    
    def __init__(
        self,
        documents: List[Document],
        k: int = 3
    ):
        # Initialize components
        self.rag = RAGSystem()
        self.retriever = self.rag.create_retriever(
            documents,
            k=k
        )
        self.qa_chain = self.rag.create_qa_chain(
            self.retriever
        )
        self.processor = ContextProcessor(
            self.rag.llm
        )
    
    async def answer_question(
        self,
        question: str
    ) -> dict:
        """
        Answer question using RAG
        
        Args:
            question: User question
            
        Returns:
            Answer and source documents
        """
        # Get answer and sources
        result = await self.qa_chain.arun(question)
        
        # Process sources
        sources = result.get("source_documents", [])
        ranked_sources = self.processor.rank_relevance(
            question,
            sources
        )
        
        return {
            "answer": result["result"],
            "sources": [
                {
                    "content": doc.page_content,
                    "relevance": score
                }
                for doc, score in ranked_sources
            ]
        }
```

### 🌐 RAG API
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Question(BaseModel):
    text: str

class RAGService:
    def __init__(self, documents):
        self.qa_system = RAGQuestionAnswering(documents)

    async def get_answer(
        self,
        question: str
    ) -> dict:
        return await self.qa_system.answer_question(
            question
        )

# Initialize service
rag_service = None

@app.post("/ask")
async def ask_question(question: Question):
    if rag_service is None:
        raise HTTPException(
            status_code=500,
            detail="Service not initialized"
        )
    
    try:
        return await rag_service.get_answer(
            question.text
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
```

## ✅ Assignments

### Level 1: Basic RAG
1. Create simple RAG system
2. Implement document retrieval
3. Add basic QA functionality

### Level 2: Advanced Features
1. Add multi-query retrieval
2. Implement context ranking
3. Create API endpoint

### Bonus Challenge
1. Add streaming responses
2. Implement caching
3. Add source validation

## 🎯 Practice Exercises

### Exercise 1: Document QA
```python
def create_document_qa():
    """
    Create system that:
    1. Retrieves relevant context
    2. Generates accurate answers
    3. Provides source citations
    """
    # Your code here
    pass
```

### Exercise 2: Context Processing
```python
def create_context_processor():
    """
    Build processor that:
    1. Ranks document relevance
    2. Summarizes context
    3. Filters irrelevant info
    """
    # Your code here
    pass
```

## 🧠 Summary
- RAG enhances LLM responses
- Retrieval strategies matter
- Context processing improves accuracy
- Source validation is crucial

## 📚 Additional Resources
1. [RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
2. [Retrieval Strategies](https://python.langchain.com/docs/modules/data_connection/retrievers/)
3. [Context Processing](https://python.langchain.com/docs/modules/chains/document/)

> **Navigation**
> - [← Vectors](38-Python-LangChain-Vectors.md)
> - [Output Parsing →](40-Python-LangChain-Output.md)
