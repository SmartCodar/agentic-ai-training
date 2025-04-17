# Day 38 – Vector Stores and Embeddings in LangChain

## Overview
Learn how to use vector stores and embeddings in LangChain for semantic search and similarity matching. This lesson covers embedding generation, vector store integration, and building search applications.

## 🎯 Learning Objectives
By the end of this lesson, you will:
- Understand vector embeddings
- Work with various vector stores
- Implement semantic search
- Build similarity-based applications

## Prerequisites
Before starting this lesson, ensure you have:
- Completed Documents lesson
- Understanding of embeddings
- Basic linear algebra knowledge
- Experience with databases

### ⚙️ Technical Requirements
- Python 3.11+
- LangChain library
- Vector store (Chroma, FAISS)
- Development environment setup

## 1. Understanding Embeddings

### 🧮 Vector Embeddings
```python
from langchain.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceEmbeddings
)

def create_embeddings(provider: str = "openai"):
    """
    Create embedding model
    
    Args:
        provider: Embedding provider
        
    Returns:
        Embedding model
    """
    if provider == "openai":
        return OpenAIEmbeddings()
    elif provider == "huggingface":
        return HuggingFaceEmbeddings()
    else:
        raise ValueError(f"Unknown provider: {provider}")
```

### 🔄 Text to Vectors
```python
from typing import List

class TextEmbedder:
    """Convert text to vector embeddings"""
    
    def __init__(self, embedding_model):
        self.embeddings = embedding_model
    
    async def embed_texts(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """
        Convert texts to embeddings
        
        Args:
            texts: List of texts
            
        Returns:
            List of embedding vectors
        """
        return await self.embeddings.aembed_documents(texts)
    
    async def embed_query(
        self,
        query: str
    ) -> List[float]:
        """
        Convert query to embedding
        """
        return await self.embeddings.aembed_query(query)
```

## 2. Vector Stores

### 💾 Store Types
```python
from langchain.vectorstores import (
    Chroma,
    FAISS,
    Pinecone
)
from langchain.docstore.document import Document

class VectorStoreManager:
    """Manage different vector stores"""
    
    def __init__(self, embedding_model):
        self.embeddings = embedding_model
    
    def create_store(
        self,
        documents: List[Document],
        store_type: str = "chroma",
        **kwargs
    ):
        """
        Create vector store
        
        Args:
            documents: Documents to store
            store_type: Type of vector store
            **kwargs: Additional arguments
            
        Returns:
            Vector store instance
        """
        if store_type == "chroma":
            return Chroma.from_documents(
                documents,
                self.embeddings,
                **kwargs
            )
        elif store_type == "faiss":
            return FAISS.from_documents(
                documents,
                self.embeddings,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown store type: {store_type}")
```

### 🔍 Similarity Search
```python
class SemanticSearch:
    """Perform semantic search operations"""
    
    def __init__(self, vector_store):
        self.store = vector_store
    
    def search(
        self,
        query: str,
        k: int = 3
    ) -> List[Document]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            Similar documents
        """
        return self.store.similarity_search(query, k=k)
    
    def search_with_scores(
        self,
        query: str,
        k: int = 3
    ) -> List[tuple]:
        """
        Search with similarity scores
        """
        return self.store.similarity_search_with_score(
            query,
            k=k
        )
```

## 3. Advanced Vector Operations

### 📊 Vector Clustering
```python
from sklearn.cluster import KMeans
import numpy as np

class VectorClustering:
    """Cluster document vectors"""
    
    def cluster_documents(
        self,
        documents: List[Document],
        embeddings,
        n_clusters: int = 5
    ):
        """
        Cluster documents by similarity
        
        Args:
            documents: Documents to cluster
            embeddings: Embedding model
            n_clusters: Number of clusters
            
        Returns:
            Cluster assignments
        """
        # Get embeddings
        vectors = embeddings.embed_documents(
            [doc.page_content for doc in documents]
        )
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(vectors)
        
        # Group documents by cluster
        clustered_docs = {}
        for i, cluster in enumerate(clusters):
            if cluster not in clustered_docs:
                clustered_docs[cluster] = []
            clustered_docs[cluster].append(documents[i])
        
        return clustered_docs
```

### 🔄 Vector Operations
```python
class VectorOperations:
    """Perform vector operations"""
    
    def vector_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """
        Calculate cosine similarity
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2)
    
    def average_vector(
        self,
        vectors: List[List[float]]
    ) -> List[float]:
        """
        Calculate average vector
        """
        return [
            sum(x) / len(vectors)
            for x in zip(*vectors)
        ]
```

## 4. Building Search Applications

### 🔍 Document Search System
```python
class DocumentSearchSystem:
    """Complete document search system"""
    
    def __init__(
        self,
        embedding_model,
        vector_store
    ):
        self.embedder = TextEmbedder(embedding_model)
        self.store = vector_store
        self.search = SemanticSearch(vector_store)
    
    async def add_documents(
        self,
        documents: List[Document]
    ):
        """Add documents to search system"""
        texts = [doc.page_content for doc in documents]
        embeddings = await self.embedder.embed_texts(texts)
        self.store.add_embeddings(texts, embeddings)
    
    async def search_documents(
        self,
        query: str,
        k: int = 3
    ) -> List[Document]:
        """Search for documents"""
        return await self.search.search(query, k)
```

### 📱 API Integration
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class SearchQuery(BaseModel):
    query: str
    k: int = 3

class SearchSystem:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.store = Chroma(
            embedding_function=self.embeddings
        )
        self.search = SemanticSearch(self.store)

search_system = SearchSystem()

@app.post("/search")
async def search_documents(query: SearchQuery):
    try:
        results = await search_system.search.search(
            query.query,
            k=query.k
        )
        return {
            "results": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
```

## ✅ Assignments

### Level 1: Basic Vector Operations
1. Create embedding system
2. Implement similarity search
3. Add basic clustering

### Level 2: Advanced Features
1. Create custom vector store
2. Implement vector operations
3. Add search API

### Bonus Challenge
1. Build multi-modal search
2. Add vector compression
3. Implement caching

## 🎯 Practice Exercises

### Exercise 1: Document Clustering
```python
def cluster_documents():
    """
    Create system that:
    1. Embeds documents
    2. Clusters by similarity
    3. Provides cluster insights
    """
    # Your code here
    pass
```

### Exercise 2: Search API
```python
def create_search_api():
    """
    Build API that:
    1. Accepts search queries
    2. Returns relevant documents
    3. Provides similarity scores
    """
    # Your code here
    pass
```

## 🧠 Summary
- Vector stores enable semantic search
- Embeddings capture meaning
- Clustering reveals patterns
- APIs make search accessible

## 📚 Additional Resources
1. [Vector Stores Guide](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
2. [Embeddings Overview](https://python.langchain.com/docs/modules/data_connection/text_embedding/)
3. [Search Systems](https://python.langchain.com/docs/use_cases/question_answering/)

> **Navigation**
> - [← Documents](37-Python-LangChain-Documents.md)
> - [Retrieval →](39-Python-LangChain-Retrieval.md)
