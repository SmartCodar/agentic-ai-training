# Day - 7 Testing and Debugging in Python

## Overview
This lesson covers essential testing and debugging techniques in Python. You'll learn how to write and run tests, debug code effectively, handle exceptions, and use logging for better error tracking. These skills are crucial for developing reliable and maintainable software.

## Learning Objectives
- Master unit testing with pytest
- Understand debugging techniques and tools
- Learn effective error handling
- Implement logging for better debugging
- Practice test-driven development (TDD)

## Prerequisites
- Understanding of Python variables and data types
- Knowledge of flow control (if statements, loops)
- Understanding of functions and their usage
- Familiarity with Python modules and imports
- Python 3.x installed on your computer

## Time Estimate
- Reading: 35 minutes
- Practice: 55 minutes
- Assignments: 45 minutes

---

## 1. Unit Testing with pytest

### Basic Test Structure
```python
# test_calculator.py
import pytest
from calculator import Calculator

def test_addition():
    calc = Calculator()
    assert calc.add(2, 3) == 5
    assert calc.add(-1, 1) == 0
    assert calc.add(0, 0) == 0

def test_division():
    calc = Calculator()
    assert calc.divide(6, 2) == 3
    assert calc.divide(5, 2) == 2.5
    
    with pytest.raises(ValueError):
        calc.divide(1, 0)  # Should raise ValueError
```

### Test Fixtures
```python
# test_database.py
import pytest
from database import Database

@pytest.fixture
def db():
    """Create a test database connection."""
    db = Database('test.db')
    db.connect()
    yield db
    db.disconnect()  # Cleanup after test

def test_insert(db):
    """Test database insertion."""
    result = db.insert('users', {'name': 'John', 'age': 30})
    assert result.success
    assert db.count('users') == 1

def test_query(db):
    """Test database query."""
    db.insert('users', {'name': 'Alice', 'age': 25})
    result = db.query('users', {'age': 25})
    assert len(result) == 1
    assert result[0]['name'] == 'Alice'
```

### Parameterized Tests
```python
# test_validation.py
import pytest
from validation import validate_email

@pytest.mark.parametrize("email,expected", [
    ("user@example.com", True),
    ("invalid.email", False),
    ("user@domain", False),
    ("user.name@company.com", True),
    ("", False),
])
def test_email_validation(email, expected):
    assert validate_email(email) == expected
```

## 2. Debugging Techniques

### Using pdb (Python Debugger)
```python
def complex_calculation(data):
    import pdb; pdb.set_trace()  # Start debugger
    result = 0
    for item in data:
        # Process each item
        if isinstance(item, (int, float)):
            result += item
        elif isinstance(item, str):
            try:
                result += float(item)
            except ValueError:
                print(f"Skipping invalid number: {item}")
    return result

# Common pdb commands:
# n (next line)
# s (step into function)
# c (continue execution)
# p variable (print variable)
# l (list source code)
# q (quit debugger)
```

### Using breakpoint()
```python
def process_data(items):
    results = []
    for item in items:
        breakpoint()  # Python 3.7+ debugging
        processed = item.strip().lower()
        if processed:
            results.append(processed)
    return results
```

## 3. Exception Handling

### Try-Except Patterns
```python
def safe_operation(data):
    try:
        # Attempt risky operation
        result = process_data(data)
        return result
    except ValueError as e:
        # Handle specific exception
        logging.error(f"Invalid data: {e}")
        return None
    except Exception as e:
        # Handle any other exception
        logging.exception("Unexpected error occurred")
        raise  # Re-raise the exception
    else:
        # Execute if no exception occurred
        logging.info("Operation completed successfully")
    finally:
        # Always execute this block
        cleanup_resources()
```

### Custom Exceptions
```python
class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, message, field=None):
        self.message = message
        self.field = field
        super().__init__(self.message)

class DataProcessor:
    def validate_input(self, data):
        if not data:
            raise ValidationError("Data cannot be empty")
        
        if 'id' not in data:
            raise ValidationError("Missing ID field", field='id')
        
        if not isinstance(data['id'], int):
            raise ValidationError("ID must be an integer", field='id')
```

## 4. Logging

### Basic Logging Setup
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def process_order(order):
    logger.info(f"Processing order {order.id}")
    try:
        # Process the order
        result = order.process()
        logger.debug(f"Order details: {order.details}")
        return result
    except Exception as e:
        logger.error(f"Failed to process order {order.id}: {str(e)}")
        raise
```

### Advanced Logging
```python
import logging.config
import yaml

def setup_logging():
    """Setup logging configuration."""
    with open('logging_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config)

class OrderProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_batch(self, orders):
        self.logger.info(f"Processing batch of {len(orders)} orders")
        
        for order in orders:
            try:
                self.process_single_order(order)
            except Exception:
                self.logger.exception(f"Error processing order {order.id}")
    
    def process_single_order(self, order):
        self.logger.debug(f"Processing order: {order.id}")
        # Process order logic here
```

## 5. Test-Driven Development (TDD)

### TDD Workflow Example
```python
# Step 1: Write the test first
def test_user_registration():
    user_service = UserService()
    
    # Test valid registration
    result = user_service.register(
        username="john_doe",
        email="john@example.com",
        password="secure123"
    )
    assert result.success
    assert result.user.username == "john_doe"
    
    # Test duplicate username
    with pytest.raises(ValidationError) as exc:
        user_service.register(
            username="john_doe",
            email="another@example.com",
            password="password123"
        )
    assert "Username already exists" in str(exc.value)

# Step 2: Implement the feature
class UserService:
    def __init__(self):
        self.users = {}
    
    def register(self, username: str, email: str, password: str) -> RegistrationResult:
        # Validate input
        if username in self.users:
            raise ValidationError("Username already exists")
        
        # Create user
        user = User(username, email, password)
        self.users[username] = user
        
        return RegistrationResult(success=True, user=user)

# Step 3: Refactor if needed
class RegistrationResult:
    def __init__(self, success: bool, user: User = None, error: str = None):
        self.success = success
        self.user = user
        self.error = error
```

## 6. Real-world Testing Example

### E-commerce System Tests
```python
import pytest
from datetime import datetime
from ecommerce import Product, Cart, Order, PaymentProcessor

class TestEcommerceSystem:
    @pytest.fixture
    def product_catalog(self):
        return [
            Product("1", "Laptop", 999.99),
            Product("2", "Mouse", 29.99),
            Product("3", "Keyboard", 59.99)
        ]
    
    @pytest.fixture
    def cart(self):
        return Cart()
    
    def test_add_to_cart(self, product_catalog, cart):
        # Add products to cart
        cart.add_item(product_catalog[0], 1)
        cart.add_item(product_catalog[1], 2)
        
        assert len(cart.items) == 2
        assert cart.total == 1059.97
    
    def test_remove_from_cart(self, product_catalog, cart):
        # Add and remove products
        cart.add_item(product_catalog[0], 1)
        cart.remove_item(product_catalog[0])
        
        assert len(cart.items) == 0
        assert cart.total == 0
    
    def test_checkout_process(self, product_catalog, cart):
        # Setup
        cart.add_item(product_catalog[0], 1)
        payment_processor = PaymentProcessor()
        
        # Process order
        order = Order(cart, "customer@example.com")
        payment_result = payment_processor.process_payment(
            order,
            card_number="4111111111111111",
            expiry="12/25",
            cvv="123"
        )
        
        assert payment_result.success
        assert order.status == "Paid"
        assert order.payment_date is not None
    
    def test_invalid_payment(self, product_catalog, cart):
        # Setup
        cart.add_item(product_catalog[0], 1)
        payment_processor = PaymentProcessor()
        
        # Test invalid card
        with pytest.raises(PaymentError) as exc:
            order = Order(cart, "customer@example.com")
            payment_processor.process_payment(
                order,
                card_number="1111111111111111",  # Invalid card
                expiry="12/25",
                cvv="123"
            )
        
        assert "Invalid card number" in str(exc.value)
        assert order.status == "Pending"
```

## 7. Best Practices

### Testing Best Practices
- Write tests before code (TDD)
- Keep tests simple and focused
- Use meaningful test names
- Test edge cases and error conditions
- Maintain test independence
- Use appropriate assertions

### Debugging Best Practices
- Use meaningful variable names
- Add strategic debug points
- Log important information
- Handle errors appropriately
- Document debugging processes

### Error Handling Best Practices
- Be specific with exceptions
- Clean up resources properly
- Log errors with context
- Fail fast and explicitly
- Don't catch generic exceptions

## 8. Summary

### Key Takeaways
- Testing ensures code reliability
- Debugging helps find and fix issues
- Proper error handling improves robustness
- Logging aids in troubleshooting
- TDD leads to better design

### What's Next
- Web Development with Python
- Working with Databases
- API Development

---

> **Navigation**
> - [← Object-Oriented Programming](05-Python-OOP.md)
> - [Web Development →](07-Python-Web-Development.md)
