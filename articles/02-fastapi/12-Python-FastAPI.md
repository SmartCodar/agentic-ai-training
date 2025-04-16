# Day - 12 Building Professional APIs with FastAPI

## Overview
This lesson teaches how to build production-ready APIs using FastAPI. You'll learn how to create robust, type-safe, and well-documented APIs with real-world examples and best practices.

## Learning Objectives
- Master FastAPI setup and configuration
- Create CRUD endpoints with validation
- Implement authentication and authorization
- Use dependency injection
- Handle database operations
- Document APIs effectively
- Implement logging and monitoring

## Prerequisites
- Understanding of async/await in Python
- Knowledge of HTTP methods and REST APIs
- Familiarity with web development concepts
- Basic command line knowledge

### Technical Requirements
- Python 3.11+ installed ([Download Python](https://www.python.org/downloads/))
- pip (Python package installer)
- Basic understanding of HTTP and REST APIs
- Familiarity with async/await syntax
- Terminal/Command Prompt experience

## Time Estimate
- Reading: 45 minutes
- Practice: 60 minutes
- Project: 90 minutes

## ⚙️ Setup Check
```python
import sys
import asyncio
from typing import Dict

async def check_fastapi_setup():
    try:
        # Check Python version
        print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
        if sys.version_info < (3, 11):
            print("Error: Python 3.11+ required")
            return False
            
        # Import and check FastAPI
        import fastapi
        print(f"FastAPI version: {fastapi.__version__}")
        
        # Check Uvicorn
        import uvicorn
        print(f"Uvicorn version: {uvicorn.__version__}")
        
        # Check Pydantic
        import pydantic
        print(f"Pydantic version: {pydantic.__version__}")
        
        # Create test app
        app = fastapi.FastAPI(title="Setup Test")
        
        @app.get("/")
        async def root() -> Dict[str, str]:
            return {"status": "ok"}
            
        print("FastAPI test app created successfully")
        return True
        
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install requirements:")
        print("pip install fastapi uvicorn[standard] pydantic")
        return False
    except Exception as e:
        print(f"Setup check failed: {e}")
        return False

# Run setup check
if __name__ == "__main__":
    asyncio.run(check_fastapi_setup())
```

---

## 1. Project Setup

### Installation and Structure
```bash
# Create project directory
mkdir fastapi_project
cd fastapi_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi[all] sqlalchemy python-jose[cryptography] passlib[bcrypt] python-multipart

# Create project structure
mkdir app app/api app/core app/models app/schemas app/services app/tests
touch app/__init__.py app/main.py
```

### Project Structure
```plaintext
fastapi_project/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── endpoints/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py
│   │   │   ├── users.py
│   │   │   └── products.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── security.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── models.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── schemas.py
│   └── services/
│       ├── __init__.py
│       └── user_service.py
├── requirements.txt
└── README.md
```

## 2. Core Configuration

### Environment Configuration (app/core/config.py)
```python
from pydantic import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    APP_NAME: str = "FastAPI E-commerce API"
    DEBUG: bool = False
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = "sqlite:///./test.db"
    
    # Security
    SECRET_KEY: str = "your-secret-key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    BACKEND_CORS_ORIGINS: list = ["*"]
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
```

### Security Setup (app/core/security.py)
```python
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from .config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)
```

## 3. Database Models and Schemas

### SQLAlchemy Models (app/models/models.py)
```python
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    products = relationship("Product", back_populates="owner")

class Product(Base):
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String)
    price = Column(Integer)  # Price in cents
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    owner = relationship("User", back_populates="products")
```

### Pydantic Schemas (app/schemas/schemas.py)
```python
from pydantic import BaseModel, EmailStr, validator
from typing import Optional, List
from datetime import datetime

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str
    
    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        orm_mode = True

class ProductBase(BaseModel):
    title: str
    description: Optional[str] = None
    price: int

class ProductCreate(ProductBase):
    pass

class Product(ProductBase):
    id: int
    owner_id: int
    created_at: datetime
    
    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
```

## 4. Service Layer

### User Service (app/services/user_service.py)
```python
from typing import Optional, List
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from ..models.models import User, Product
from ..schemas.schemas import UserCreate, ProductCreate
from ..core.security import get_password_hash, verify_password

class UserService:
    def __init__(self, db: Session):
        self.db = db
    
    def get_user(self, user_id: int) -> Optional[User]:
        return self.db.query(User).filter(User.id == user_id).first()
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        return self.db.query(User).filter(User.email == email).first()
    
    def get_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        return self.db.query(User).offset(skip).limit(limit).all()
    
    def create_user(self, user: UserCreate) -> User:
        if self.get_user_by_email(user.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        db_user = User(
            email=user.email,
            hashed_password=get_password_hash(user.password)
        )
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        return db_user
    
    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        user = self.get_user_by_email(email)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user
    
    def create_user_product(
        self, user_id: int, product: ProductCreate
    ) -> Product:
        db_product = Product(**product.dict(), owner_id=user_id)
        self.db.add(db_product)
        self.db.commit()
        self.db.refresh(db_product)
        return db_product
```

## 5. API Endpoints

### Authentication Endpoints (app/api/endpoints/auth.py)
```python
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import Any
from ...core.security import create_access_token
from ...services.user_service import UserService
from ...schemas.schemas import Token
from ...core.config import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()) -> Token:
    # Add implementation
    pass
```

### Exercise 3: Database Integration
Connect to a database using SQLAlchemy:
```python
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

async def get_user(db: AsyncSession, user_id: int) -> Optional[User]:
    # Add implementation
    pass
```

## 7. Knowledge Check ✅

1. What are the key features of FastAPI?
2. How do you implement path parameters and query parameters?
3. What is dependency injection and how is it used in FastAPI?
4. How do you handle authentication in FastAPI?
5. What is Pydantic and how does it help with data validation?
6. How do you implement database operations with FastAPI?
7. What are FastAPI's best practices for error handling?
8. How do you document APIs using FastAPI's automatic documentation?

## 8. Summary

### Key Takeaways
1. FastAPI provides modern, fast API development
2. Type hints enable automatic validation
3. Async support improves performance
4. Built-in documentation with OpenAPI
5. Security features are production-ready

## 📚 Additional Resources
- [FastAPI Official Documentation](https://fastapi.tiangolo.com/)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/advanced/)
- [Real Python - FastAPI Tutorial](https://realpython.com/fastapi-python-web-apis/)
- [Async Database Operations](https://fastapi.tiangolo.com/advanced/async-sql-databases/)

---

> **Navigation**
> - [← Aiohttp Client](11-Python-Aiohttp-Client.md)
> - [FastAPI Database →](13-Python-FastAPI-Database.md)
from ...core.deps import get_db

router = APIRouter()

@router.post("/login", response_model=Token)
async def login(
    db: Session = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Any:
    user_service = UserService(db)
    user = user_service.authenticate_user(
        form_data.username, form_data.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}
```

### User Endpoints (app/api/endpoints/users.py)
```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Any, List
from ...services.user_service import UserService
from ...schemas.schemas import User, UserCreate, Product, ProductCreate
from ...core.deps import get_current_user, get_db

router = APIRouter()

@router.post("", response_model=User)
def create_user(
    user: UserCreate,
    db: Session = Depends(get_db)
) -> Any:
    user_service = UserService(db)
    return user_service.create_user(user)

@router.get("/me", response_model=User)
def read_user_me(
    current_user: User = Depends(get_current_user)
) -> Any:
    return current_user

@router.post("/me/products", response_model=Product)
def create_product_for_user(
    product: ProductCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    user_service = UserService(db)
    return user_service.create_user_product(
        user_id=current_user.id, product=product
    )
```

## 6. Main Application

### Main Application Setup (app/main.py)
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings
from .api.endpoints import auth, users, products

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    auth.router,
    prefix=f"{settings.API_PREFIX}/auth",
    tags=["authentication"]
)
app.include_router(
    users.router,
    prefix=f"{settings.API_PREFIX}/users",
    tags=["users"]
)
app.include_router(
    products.router,
    prefix=f"{settings.API_PREFIX}/products",
    tags=["products"]
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

## 7. Running the Application

### Start the Server
```bash
# Development server
uvicorn app.main:app --reload --port 8000

# Production server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Testing Endpoints
```bash
# Create user
curl -X POST "http://localhost:8000/api/v1/users" \
     -H "Content-Type: application/json" \
     -d '{"email": "user@example.com", "password": "password123"}'

# Login
curl -X POST "http://localhost:8000/api/v1/auth/login" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=user@example.com&password=password123"

# Create product (with token)
curl -X POST "http://localhost:8000/api/v1/users/me/products" \
     -H "Authorization: Bearer <your-token>" \
     -H "Content-Type: application/json" \
     -d '{"title": "My Product", "description": "Description", "price": 1999}'
```

## 8. Best Practices

### Security
1. Always hash passwords
2. Use HTTPS in production
3. Implement rate limiting
4. Validate input data
5. Use proper authentication

### Performance
1. Use async where appropriate
2. Implement caching
3. Optimize database queries
4. Use connection pooling

### Code Organization
1. Follow repository pattern
2. Implement service layer
3. Use dependency injection
4. Keep endpoints thin

## 9. Knowledge Check ✅

1. What are the key features of FastAPI that make it different from other frameworks?
2. How do you implement path parameters and query parameters in FastAPI?
3. What is dependency injection and how is it used in FastAPI?
4. How do you handle authentication and authorization?
5. What is Pydantic and how does it help with data validation?
6. How do you implement proper error handling in FastAPI?
7. What are the best practices for organizing a FastAPI project?
8. How do you document your APIs using FastAPI's automatic documentation?

## 10. Summary

### Key Takeaways
1. FastAPI provides a modern, fast framework for APIs
2. Type hints and validation improve reliability
3. Proper project structure is crucial
4. Security must be a priority
5. Documentation is automated and comprehensive

## 📚 Additional Resources
- [FastAPI Official Documentation](https://fastapi.tiangolo.com/)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/advanced/)
- [Real Python - FastAPI Tutorial](https://realpython.com/fastapi-python-web-apis/)
- [Async Database Operations](https://fastapi.tiangolo.com/advanced/async-sql-databases/)
- [Security Best Practices](https://fastapi.tiangolo.com/tutorial/security/)

---

> **Navigation**
> - [← Aiohttp Client](11-Python-Aiohttp-Client.md)
> - [Database Integration →](13-Python-Database.md)
