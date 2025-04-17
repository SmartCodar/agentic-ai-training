# Day - 15 Building a Mini Project: Todo API

## Overview
Build a simple Todo API using FastAPI to practice the core concepts learned so far.

## Learning Objectives
- Create a basic FastAPI application
- Implement CRUD operations
- Use Pydantic models
- Handle basic error cases
- Test API endpoints

## Prerequisites
- Understanding of FastAPI basics
- Knowledge of Pydantic models
- Familiarity with HTTP methods
- Python 3.11+ installed

## 1. Project Setup

### Create Project Structure
```bash
todo_api/
├── app.py
├── models.py
└── requirements.txt
```

### Install Dependencies
```bash
pip install fastapi uvicorn[standard] pydantic
```

## 2. Implementation

### models.py
```python
from pydantic import BaseModel
from typing import Optional

class Todo(BaseModel):
    id: Optional[int] = None
    title: str
    description: Optional[str] = None
    completed: bool = False
```

### app.py
```python
from fastapi import FastAPI, HTTPException
from models import Todo
from typing import List, Dict

app = FastAPI(title="Todo API")

# In-memory storage
todos: Dict[int, Todo] = {}
counter = 0

@app.post("/todos/", response_model=Todo)
async def create_todo(todo: Todo) -> Todo:
    global counter
    counter += 1
    todo.id = counter
    todos[todo.id] = todo
    return todo

@app.get("/todos/", response_model=List[Todo])
async def list_todos() -> List[Todo]:
    return list(todos.values())

@app.get("/todos/{todo_id}", response_model=Todo)
async def get_todo(todo_id: int) -> Todo:
    if todo_id not in todos:
        raise HTTPException(status_code=404, detail="Todo not found")
    return todos[todo_id]

@app.put("/todos/{todo_id}", response_model=Todo)
async def update_todo(todo_id: int, todo: Todo) -> Todo:
    if todo_id not in todos:
        raise HTTPException(status_code=404, detail="Todo not found")
    todo.id = todo_id
    todos[todo_id] = todo
    return todo

@app.delete("/todos/{todo_id}")
async def delete_todo(todo_id: int) -> Dict[str, bool]:
    if todo_id not in todos:
        raise HTTPException(status_code=404, detail="Todo not found")
    del todos[todo_id]
    return {"success": True}
```

## 3. Running the API
```bash
uvicorn app:app --reload
```

## 4. Testing Endpoints

### Create Todo
```bash
curl -X POST "http://localhost:8000/todos/" \
     -H "Content-Type: application/json" \
     -d '{"title": "Learn FastAPI", "description": "Complete the tutorial"}'
```

### List Todos
```bash
curl "http://localhost:8000/todos/"
```

## Summary

### Key Takeaways
1. FastAPI simplifies API development
2. Pydantic ensures data validation
3. CRUD operations are straightforward
4. Error handling improves reliability
5. API documentation is automatic

## ✅ Knowledge Check
1. How do you create a new FastAPI application?
2. What is the purpose of response_model?
3. How do you handle path parameters?
4. How do you implement error handling?
5. What are the benefits of using Pydantic models?
6. How do you test FastAPI endpoints?
7. What is the difference between PUT and POST?
8. How do you access the API documentation?

## 📚 Additional Resources
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Pydantic Models](https://docs.pydantic.dev/)
- [HTTP Methods](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods)
- [REST API Best Practices](https://fastapi.tiangolo.com/tutorial/metadata/)
- [Testing FastAPI](https://fastapi.tiangolo.com/tutorial/testing/)

---

> **Navigation**
> - [← Pydantic](14-Python-Pydantic.md)
> - [FastAPI Project →](16-Python-FastAPI-Project.md)
