# Day - 14 Pydantic: Advanced Data Validation and Schema Management

## Overview
Master Pydantic's powerful data validation and schema management capabilities for building robust APIs and data-driven applications.

## Learning Objectives
- Understand Pydantic's core concepts
- Implement complex data validation
- Create nested and recursive models
- Use custom validators and types
- Handle data serialization/deserialization
- Apply best practices for schema design

## Prerequisites
- Understanding of FastAPI basics and routing
- Knowledge of Python data types and type hints
- Familiarity with data validation concepts
- Understanding of Python classes and inheritance
- Python 3.7+ installed
- FastAPI and Pydantic installed
- Understanding of Python type hints
- Knowledge of data validation concepts
- Basic understanding of JSON schemas

## Time Estimate
- Reading: 30 minutes
- Practice: 45 minutes
- Exercises: 45 minutes

---

## 1. Pydantic Basics

### Core Concepts
```python
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime

class User(BaseModel):
    id: int
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "username": "johndoe",
                "email": "john@example.com",
                "full_name": "John Doe",
                "created_at": "2025-04-15T10:00:00"
            }
        }
```

## 2. Advanced Field Validation

### Complex Constraints
```python
from pydantic import BaseModel, Field, validator, constr
from typing import List, Optional
from decimal import Decimal

class Product(BaseModel):
    id: int
    name: constr(min_length=3, max_length=50)  # Constrained string
    price: Decimal = Field(..., ge=0, decimal_places=2)
    description: Optional[str] = Field(None, max_length=1000)
    tags: List[str] = Field(default_factory=list, max_items=10)
    
    @validator('tags')
    def validate_tags(cls, v):
        if not all(isinstance(tag, str) and tag.strip() for tag in v):
            raise ValueError("All tags must be non-empty strings")
        return [tag.lower() for tag in v]  # Normalize tags to lowercase
```

### Custom Types and Validators
```python
from pydantic import BaseModel, validator, conint
from typing import Optional
from enum import Enum
import re

class PaymentStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"

class CreditCard(BaseModel):
    number: str
    expiry_month: conint(ge=1, le=12)  # Constrained integer
    expiry_year: conint(ge=2025, le=2100)
    cvv: str
    
    @validator('number')
    def validate_card_number(cls, v):
        # Remove spaces and dashes
        v = re.sub(r'[\s-]', '', v)
        if not v.isdigit() or len(v) not in [15, 16]:
            raise ValueError("Invalid card number")
        return v
    
    @validator('cvv')
    def validate_cvv(cls, v):
        if not v.isdigit() or len(v) not in [3, 4]:
            raise ValueError("Invalid CVV")
        return v

class Payment(BaseModel):
    amount: Decimal = Field(..., ge=0.01, decimal_places=2)
    currency: str = Field(..., regex='^[A-Z]{3}$')  # ISO currency code
    status: PaymentStatus = PaymentStatus.PENDING
    card: CreditCard
    description: Optional[str] = None
```

## 3. Nested Models and Relationships

### Complex Data Structures
```python
from pydantic import BaseModel, Field, root_validator
from typing import Dict, List, Optional
from datetime import datetime

class Address(BaseModel):
    street: str
    city: str
    state: str
    postal_code: str
    country: str = Field(..., max_length=2)  # ISO country code

class OrderItem(BaseModel):
    product_id: int
    quantity: int = Field(..., gt=0)
    unit_price: Decimal
    
    @property
    def total_price(self) -> Decimal:
        return self.quantity * self.unit_price

class Order(BaseModel):
    id: int
    user_id: int
    items: List[OrderItem]
    shipping_address: Address
    billing_address: Optional[Address]
    status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @root_validator
    def validate_addresses(cls, values):
        shipping = values.get('shipping_address')
        billing = values.get('billing_address')
        if billing is None:
            values['billing_address'] = shipping
        return values
    
    @property
    def total_amount(self) -> Decimal:
        return sum(item.total_price for item in self.items)
```

## 4. Data Serialization and Deserialization

### Custom Serialization
```python
from pydantic import BaseModel
from datetime import datetime
import json

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)

class APIResponse(BaseModel):
    success: bool
    data: Dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    def json(self, **kwargs):
        return json.dumps(
            self.dict(),
            cls=CustomJSONEncoder,
            **kwargs
        )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: str
        }
```

## 5. Advanced Use Cases

### Dynamic Model Generation
```python
from pydantic import create_model, Field
from typing import Dict, Any

def create_dynamic_model(fields: Dict[str, Any]) -> type[BaseModel]:
    """Create a Pydantic model dynamically based on field definitions."""
    field_definitions = {}
    
    for field_name, field_type in fields.items():
        if isinstance(field_type, tuple):
            field_type, field_config = field_type
        else:
            field_config = {}
            
        field_definitions[field_name] = (field_type, Field(**field_config))
    
    return create_model('DynamicModel', **field_definitions)

# Usage example
fields = {
    'name': (str, {'min_length': 3}),
    'age': (int, {'ge': 0}),
    'email': (EmailStr, {}),
}

UserModel = create_dynamic_model(fields)
```

### Schema Evolution and Versioning
```python
from pydantic import BaseModel, Field
from typing import Optional, Union

class UserV1(BaseModel):
    username: str
    email: str

class UserV2(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None

class UserV3(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)

def parse_user(data: Dict[str, Any]) -> Union[UserV1, UserV2, UserV3]:
    """Parse user data based on version information."""
    version = data.get('version', 1)
    
    if version == 1:
        return UserV1(**data)
    elif version == 2:
        return UserV2(**data)
    else:
        return UserV3(**data)
```

## 6. Best Practices

### 1. Model Organization
```python
# models/base.py
from pydantic import BaseModel, Field
from datetime import datetime

class TimestampedModel(BaseModel):
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

# models/user.py
from .base import TimestampedModel

class User(TimestampedModel):
    username: str
    email: EmailStr
```

### 2. Validation Helpers
```python
from pydantic import BaseModel, validator
import re

class ValidationMixin:
    @classmethod
    def validate_phone(cls, v: str) -> str:
        if not re.match(r'^\+?1?\d{9,15}$', v):
            raise ValueError("Invalid phone number")
        return v

class Contact(BaseModel, ValidationMixin):
    phone: str
    
    _validate_phone = validator('phone', allow_reuse=True)(
        ValidationMixin.validate_phone
    )
```

## Summary

### Key Takeaways
1. Pydantic provides robust data validation
2. Custom validators enhance data integrity
3. Nested models handle complex structures
4. Schema evolution requires careful planning

### Best Practices
1. Use type hints consistently
2. Implement custom validators for complex logic
3. Plan for schema evolution
4. Keep models organized and maintainable

---

> **Navigation**
> - [← FastAPI Routes](13-Python-FastAPI-Routes.md)
> - [FastAPI Project →](15-Python-FastAPI-Project.md)
