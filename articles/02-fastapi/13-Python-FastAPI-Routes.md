# Day - 13 FastAPI Routes and Advanced Request Handling

## Overview
Learn how to create professional API endpoints using FastAPI's powerful routing system, parameter handling, and request validation features.

## Learning Objectives
- Master path parameters and type validation
- Implement query parameters effectively
- Handle request bodies with Pydantic models
- Create complex routing patterns
- Implement proper error handling
- Use dependency injection in routes

## Prerequisites
- Understanding of FastAPI basics
- Knowledge of HTTP methods (GET, POST, etc.)
- Familiarity with API endpoints and routing
- Understanding of request/response cycles
- Python 3.7+ installed
- Understanding of HTTP methods (GET, POST, PUT, DELETE)
- Knowledge of path and query parameters
- Basic understanding of request/response cycle

## Time Estimate
- Reading: 30 minutes
- Practice: 45 minutes
- Exercises: 45 minutes

---

## 1. Path Parameters

### Basic Path Parameters
```python
from fastapi import FastAPI, HTTPException
from typing import Optional

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    # Type validation happens automatically!
    if user_id > 100:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user_id": user_id, "name": f"User {user_id}"}

@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    # :path allows for / in the parameter
    return {"file_path": file_path}
```

### Enum Path Parameters
```python
from enum import Enum
from fastapi import FastAPI

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

app = FastAPI()

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name == ModelName.alexnet:
        return {"model": model_name, "message": "Deep Learning FTW!"}
    
    if model_name.value == "lenet":
        return {"model": model_name, "message": "LeCNN all the images"}
        
    return {"model": model_name, "message": "Have some residuals"}
```

## 2. Query Parameters

### Optional and Default Values
```python
from fastapi import FastAPI, Query
from typing import Optional, List

app = FastAPI()

@app.get("/items/")
async def read_items(
    skip: int = 0,
    limit: int = 10,
    q: Optional[str] = None,
    tags: List[str] = Query(None)
):
    items = [{"id": i, "name": f"Item {i}"} for i in range(skip, skip + limit)]
    
    if q:
        items = [item for item in items if q.lower() in item["name"].lower()]
    
    if tags:
        # Filter by tags (assuming items had tags)
        return {"items": items, "tags": tags}
        
    return {"items": items}
```

### Validation with Query Parameters
```python
from fastapi import FastAPI, Query
from typing import Optional

app = FastAPI()

@app.get("/products/")
async def read_products(
    q: Optional[str] = Query(
        None,
        min_length=3,
        max_length=50,
        regex="^[a-zA-Z0-9-]*$",
        title="Search Query",
        description="Search query string"
    ),
    category: Optional[str] = Query(
        None,
        alias="cat"  # Allow using ?cat=electronics instead of ?category=electronics
    ),
    price_min: Optional[float] = Query(None, ge=0),
    price_max: Optional[float] = Query(None, le=1000)
):
    filters = {"q": q, "category": category}
    if price_min is not None:
        filters["price_min"] = price_min
    if price_max is not None:
        filters["price_max"] = price_max
    
    return {"filters": filters}
```

## 3. Request Body

### Basic Request Body
```python
from fastapi import FastAPI
from pydantic import BaseModel, EmailStr
from typing import Optional

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    password: str

app = FastAPI()

@app.post("/users/")
async def create_user(user: UserCreate):
    # Pydantic handles validation automatically
    return {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name
    }
```

### Multiple Body Parameters
```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

class User(BaseModel):
    username: str
    full_name: Optional[str] = None

app = FastAPI()

@app.post("/items/{item_id}")
async def create_item(
    item_id: int,
    item: Item,
    user: User,
    importance: Optional[int] = None
):
    result = {
        "item_id": item_id,
        "item": item.dict(),
        "user": user.dict()
    }
    if importance:
        result["importance"] = importance
    return result
```

## 4. Form Data and File Uploads

### Form Data
```python
from fastapi import FastAPI, Form, File, UploadFile
from typing import List

app = FastAPI()

@app.post("/login/")
async def login(
    username: str = Form(...),
    password: str = Form(...),
    remember_me: bool = Form(False)
):
    return {
        "username": username,
        "remember_me": remember_me
    }

@app.post("/files/")
async def create_files(
    files: List[UploadFile] = File(...),
    description: str = Form(None)
):
    return {
        "filenames": [file.filename for file in files],
        "description": description
    }
```

## 5. Response Models

### Custom Response Models
```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

class ItemBase(BaseModel):
    name: str
    price: float
    description: Optional[str] = None

class ItemCreate(ItemBase):
    pass

class Item(ItemBase):
    id: int
    owner_id: int

    class Config:
        orm_mode = True

app = FastAPI()

@app.post("/items/", response_model=Item)
async def create_item(item: ItemCreate):
    # Simulate database creation
    db_item = {
        **item.dict(),
        "id": 1,
        "owner_id": 1
    }
    return db_item

@app.get("/items/", response_model=List[Item])
async def read_items():
    # Simulate database query
    items = [
        {
            "name": "Foo",
            "price": 50.2,
            "description": "The Foo item",
            "id": 1,
            "owner_id": 1
        },
        {
            "name": "Bar",
            "price": 62.0,
            "id": 2,
            "owner_id": 1
        }
    ]
    return items
```

## 6. Error Handling

### Custom Exception Handlers
```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.requests import Request

app = FastAPI()

class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=418,
        content={
            "message": f"Oops! {exc.name} did something wrong.",
            "unicorn_name": exc.name
        }
    )

@app.get("/unicorns/{name}")
async def read_unicorn(name: str):
    if name == "yolo":
        raise UnicornException(name=name)
    return {"unicorn_name": name}
```

## 7. Practical Examples

### E-commerce API Endpoints
```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime

# Models
class Product(BaseModel):
    id: int
    name: str
    price: float
    description: Optional[str] = None
    stock: int

class Order(BaseModel):
    id: int
    user_id: int
    products: List[Product]
    total: float
    status: str
    created_at: datetime

# API
app = FastAPI()

# Database simulation
products_db = {}
orders_db = {}

@app.get("/products/", response_model=List[Product])
async def get_products(
    skip: int = 0,
    limit: int = 10,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None
):
    products = list(products_db.values())
    
    if min_price is not None:
        products = [p for p in products if p.price >= min_price]
    if max_price is not None:
        products = [p for p in products if p.price <= max_price]
        
    return products[skip : skip + limit]

@app.post("/orders/", response_model=Order)
async def create_order(
    products: List[int],  # List of product IDs
    user_id: int
):
    if not products:
        raise HTTPException(
            status_code=400,
            detail="No products provided"
        )
    
    # Calculate total and check stock
    total = 0
    order_products = []
    for product_id in products:
        if product_id not in products_db:
            raise HTTPException(
                status_code=404,
                detail=f"Product {product_id} not found"
            )
        
        product = products_db[product_id]
        if product.stock <= 0:
            raise HTTPException(
                status_code=400,
                detail=f"Product {product.name} out of stock"
            )
            
        total += product.price
        order_products.append(product)
        
    # Create order
    order = Order(
        id=len(orders_db) + 1,
        user_id=user_id,
        products=order_products,
        total=total,
        status="pending",
        created_at=datetime.utcnow()
    )
    
    orders_db[order.id] = order
    return order
```

## Summary

### Key Takeaways
1. FastAPI provides powerful parameter validation
2. Request bodies can be validated using Pydantic models
3. Response models ensure consistent API output
4. Error handling can be customized for better UX

### Best Practices
1. Always validate input data
2. Use appropriate HTTP methods
3. Implement proper error handling
4. Document your API endpoints
5. Use response models for type safety

## ⚙️ Setup Check
```python
import sys
import fastapi

def check_setup():
    # Check Python version
    print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
    if sys.version_info < (3, 11):
        print("Error: Python 3.11+ required")
        return False
        
    # Check FastAPI
    print(f"FastAPI version: {fastapi.__version__}")
    
    # Test route parameters
    from fastapi import Path, Query
    print("Route parameter tools available")
    
    return True

if __name__ == "__main__":
    check_setup()
```

## ✅ Knowledge Check
1. What is the difference between path parameters and query parameters?
2. How do you implement optional path parameters?
3. What are the benefits of using Pydantic models for request bodies?
4. How do you handle file uploads in FastAPI?
5. What is the purpose of response_model in route decorators?
6. How do you implement proper error handling in routes?
7. What are the best practices for route organization?
8. How do you use dependency injection in routes?

## 📚 Additional Resources
- [FastAPI Path Parameters](https://fastapi.tiangolo.com/tutorial/path-params/)
- [Query Parameters](https://fastapi.tiangolo.com/tutorial/query-params/)
- [Request Files](https://fastapi.tiangolo.com/tutorial/request-files/)
- [Dependencies in Path Operations](https://fastapi.tiangolo.com/tutorial/dependencies/)
- [Response Models](https://fastapi.tiangolo.com/tutorial/response-model/)

---

> **Navigation**
> - [← FastAPI Basics](12-Python-FastAPI.md)
> - [Pydantic →](14-Python-Pydantic.md)
