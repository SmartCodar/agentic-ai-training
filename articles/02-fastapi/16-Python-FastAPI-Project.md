# Day - 16 Building a Mini Project: User Management API with CORS

## Overview
Build a simple but complete User Management API that demonstrates FastAPI features including CORS, async operations, and proper project structure.

## Learning Objectives
- Create a complete FastAPI mini-project
- Implement CORS middleware
- Use async/await properly
- Structure a small-scale project
- Implement logging

## Prerequisites
- Completion of [01 - Python Basics](../01-basics/01-Python-Basics-Variables-Types-Operators.md)
- Completion of [02 - Flow Control](../01-basics/02-Python-Flow-Control-Loops-Conditions.md)
- Completion of [03 - Functions](../01-basics/03-Python-Functions-Modular-Programming.md)
- Completion of [04 - Modules and Packages](../01-basics/04-Python-Modules-Packages.md)
- Completion of [05 - Object-Oriented Programming](../01-basics/05-Python-OOP.md)
- Completion of [06 - File Handling](../01-basics/06-Python-File-Handling.md)
- Completion of [07 - Testing and Debugging](../01-basics/07-Python-Testing-Debugging.md)
- Completion of [08 - Functional Programming](../01-basics/08-Python-Functional-Programming.md)
- Completion of [09 - Project Setup](../01-basics/09-Python-Project-Setup.md)
- Completion of [10 - Async Programming](10-Python-Async-Programming.md)
- Completion of [11 - Aiohttp Client](11-Python-Aiohttp-Client.md)
- Completion of [12 - FastAPI Basics](12-Python-FastAPI.md)
- Completion of [13 - FastAPI Routes](13-Python-FastAPI-Routes.md)
- Completion of [14 - Pydantic](14-Python-Pydantic.md)
- Completion of [15 - FastAPI Mini Project](15-Python-FastAPI-Mini-Project.md)

## Time Estimate
- Setup: 15 minutes
- Implementation: 45 minutes
- Testing: 15 minutes

---

## 1. Project Setup

### Project Structure
```bash
user_mgmt_api/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── database.py
│   ├── schemas.py
│   └── utils.py
├── requirements.txt
├── README.md
└── .gitignore
```

### Initial Files

#### requirements.txt
```txt
fastapi==0.68.1
uvicorn==0.15.0
pydantic[email]==1.8.2
python-jose[cryptography]==3.3.0
```

#### .gitignore
```gitignore
venv/
__pycache__/
*.pyc
.env
```

## 2. Implementation

### Models (app/models.py)
```python
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class User(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    is_active: bool = True
    created_at: datetime = datetime.utcnow()

class UserCreate(BaseModel):
    email: EmailStr
    full_name: str
    password: str

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None
```

### Database Simulation (app/database.py)
```python
from typing import Dict, Optional
from .models import User

# Simulate a database with a dictionary
users_db: Dict[int, User] = {}
current_id = 1

def get_user(user_id: int) -> Optional[User]:
    return users_db.get(user_id)

def create_user(user: User) -> User:
    global current_id
    user.id = current_id
    users_db[current_id] = user
    current_id += 1
    return user

def update_user(user_id: int, user: User) -> Optional[User]:
    if user_id in users_db:
        users_db[user_id] = user
        return user
    return None

def delete_user(user_id: int) -> bool:
    if user_id in users_db:
        del users_db[user_id]
        return True
    return False

def list_users() -> list[User]:
    return list(users_db.values())
```

### Main Application (app/main.py)
```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import logging

from .models import User, UserCreate, UserUpdate
from . import database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="User Management API",
    description="A simple user management API with CORS support",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/users/", response_model=List[User])
async def list_users():
    """List all users"""
    logger.info("Listing all users")
    return database.list_users()

@app.post("/users/", response_model=User)
async def create_user(user: UserCreate):
    """Create a new user"""
    logger.info(f"Creating user with email: {user.email}")
    new_user = User(
        id=0,  # Will be set by database
        email=user.email,
        full_name=user.full_name,
        is_active=True
    )
    return database.create_user(new_user)

@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int):
    """Get a specific user by ID"""
    logger.info(f"Fetching user with ID: {user_id}")
    user = database.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.put("/users/{user_id}", response_model=User)
async def update_user(user_id: int, user_update: UserUpdate):
    """Update a user"""
    logger.info(f"Updating user with ID: {user_id}")
    current_user = database.get_user(user_id)
    if current_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update only provided fields
    update_data = user_update.dict(exclude_unset=True)
    updated_user = current_user.copy(update=update_data)
    
    result = database.update_user(user_id, updated_user)
    if result is None:
        raise HTTPException(status_code=404, detail="User not found")
    return result

@app.delete("/users/{user_id}")
async def delete_user(user_id: int):
    """Delete a user"""
    logger.info(f"Deleting user with ID: {user_id}")
    if not database.delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
```

## 3. Running the Application

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload --port 8000
```

## 4. Testing the API

### Using curl
```bash
# Create a user
curl -X POST "http://localhost:8000/users/" \
     -H "Content-Type: application/json" \
     -d '{"email": "user@example.com", "full_name": "John Doe", "password": "secret123"}'

# List users
curl "http://localhost:8000/users/"

# Get specific user
curl "http://localhost:8000/users/1"

# Update user
curl -X PUT "http://localhost:8000/users/1" \
     -H "Content-Type: application/json" \
     -d '{"full_name": "John Smith"}'

# Delete user
curl -X DELETE "http://localhost:8000/users/1"
```

## 5. Best Practices

### DOs:
1. Use async/await consistently
2. Implement proper error handling
3. Add logging for debugging
4. Structure code modularly
5. Use type hints
6. Document API endpoints

### DON'Ts:
1. Leave CORS open in production
2. Expose sensitive information
3. Skip error handling
4. Mix sync and async code

## Summary

### Key Features
1. Complete CRUD operations
2. CORS middleware
3. Async operations
4. Error handling
5. Logging
6. Type validation

### Next Steps
1. Add authentication
2. Implement real database
3. Add more validation
4. Enhance error handling

---

> **Navigation**
> - [← FastAPI Mini Project](15-Python-FastAPI-Mini-Project.md)
> - [LLM Transformers →](../03-llm/17-Python-LLM-Transformers.md)
