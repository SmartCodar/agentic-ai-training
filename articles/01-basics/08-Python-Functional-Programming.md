# Day 8: Functional Programming in Python

## Introduction
Functional programming in Python enables you to:
- Write more concise and expressive code
- Create reusable and composable functions
- Process data efficiently with built-in tools
- Implement elegant solutions to complex problems

## 🎯 Learning Objectives
By the end of this lesson, you will be able to:
- Create and use lambda functions effectively
- Build and apply function decorators
- Implement memory-efficient generators
- Use map, filter, and reduce operations
- Apply functional programming patterns
- Write pure functions and avoid side effects

## 📋 Prerequisites
- Python 3.11+ installed ([Download Python](https://www.python.org/downloads/))
- Code editor (VS Code recommended) with Python extension
- Strong understanding of functions (Day 3)
- Basic knowledge of error handling

## 1. Lambda Functions

### Understanding Lambda Functions
Lambda functions are small, anonymous functions that can have any number of arguments but can only have one expression. They are perfect for simple operations and short-lived function definitions.

### Real-world Example: Data Processing
```python
from typing import List, Dict, Callable
import json

class DataProcessor:
    def __init__(self, data: List[Dict]):
        self.data = data
    
    def process_with_operation(self, operation: Callable):
        """Process data with given operation."""
        return list(map(operation, self.data))
    
    def filter_with_condition(self, condition: Callable):
        """Filter data with given condition."""
        return list(filter(condition, self.data))

# Sample data: Sales records
sales_data = [
    {"product": "Laptop", "price": 999.99, "quantity": 5},
    {"product": "Mouse", "price": 29.99, "quantity": 10},
    {"product": "Keyboard", "price": 59.99, "quantity": 7},
    {"product": "Monitor", "price": 299.99, "quantity": 3}
]

# Create processor
processor = DataProcessor(sales_data)

# Calculate total value for each item
total_values = processor.process_with_operation(
    lambda x: {
        "product": x["product"],
        "total_value": x["price"] * x["quantity"]
    }
)

# Filter expensive items
expensive_items = processor.filter_with_condition(
    lambda x: x["price"] > 100
)

# Sort by quantity
sorted_by_quantity = sorted(
    sales_data,
    key=lambda x: x["quantity"],
    reverse=True
)

print("Total values:")
print(json.dumps(total_values, indent=2))

print("\nExpensive items:")
print(json.dumps(expensive_items, indent=2))

print("\nSorted by quantity:")
print(json.dumps(sorted_by_quantity, indent=2))
```

## 2. Decorators

### Understanding Decorators
Decorators are functions that modify the behavior of other functions. They allow you to add functionality to existing functions without modifying their source code.

### Real-world Example: Web API Authentication
```python
from functools import wraps
from typing import Callable, Dict, Optional
from datetime import datetime, timedelta
import jwt
import time

class AuthenticationError(Exception):
    pass

class APIHandler:
    def __init__(self):
        self.secret_key = "your-secret-key"
        self._tokens: Dict[str, str] = {}
    
    def generate_token(self, username: str) -> str:
        """Generate JWT token."""
        expiration = datetime.utcnow() + timedelta(hours=1)
        token = jwt.encode(
            {"username": username, "exp": expiration},
            self.secret_key,
            algorithm="HS256"
        )
        self._tokens[username] = token
        return token
    
    def verify_token(self, token: str) -> Optional[str]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=["HS256"]
            )
            return payload["username"]
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
    
    def require_auth(self, func: Callable) -> Callable:
        """Decorator for requiring authentication."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            token = kwargs.get('token')
            if not token:
                raise AuthenticationError("No token provided")
            
            username = self.verify_token(token)
            if not username:
                raise AuthenticationError("Invalid token")
            
            # Add username to kwargs
            kwargs['username'] = username
            return func(*args, **kwargs)
        return wrapper
    
    def log_execution(self, func: Callable) -> Callable:
        """Decorator for logging execution time."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            print(f"Function {func.__name__} took "
                  f"{end_time - start_time:.2f} seconds")
            return result
        return wrapper

# Example usage
api = APIHandler()

@api.require_auth
@api.log_execution
def get_user_data(user_id: int, username: str = None) -> Dict:
    """Get user data (requires authentication)."""
    # Simulate database query
    time.sleep(1)
    return {
        "user_id": user_id,
        "username": username,
        "last_access": datetime.now().isoformat()
    }

# Demonstrate decorator usage
def demonstrate_api():
    try:
        # Generate token for user
        token = api.generate_token("john_doe")
        print(f"Generated token: {token}\n")
        
        # Try accessing with valid token
        result = get_user_data(
            user_id=123,
            token=token
        )
        print("Success! User data:")
        print(json.dumps(result, indent=2))
        
        # Try accessing with invalid token
        result = get_user_data(
            user_id=123,
            token="invalid-token"
        )
        
    except AuthenticationError as e:
        print(f"Authentication error: {e}")

# Run demonstration
demonstrate_api()
```

## 3. Generators

### Understanding Generators
Generators are functions that generate a sequence of values over time, rather than computing them all at once. They are memory efficient and perfect for handling large datasets.

### Real-world Example: Log File Processing
```python
from typing import Generator, Dict, List
from datetime import datetime
import re
from collections import defaultdict

class LogAnalyzer:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.pattern = re.compile(
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) '
            r'\[(\w+)\] (.*)'
        )
    
    def parse_log_lines(self) -> Generator[Dict, None, None]:
        """Generate parsed log entries."""
        with open(self.log_file, 'r') as file:
            for line in file:
                match = self.pattern.match(line.strip())
                if match:
                    timestamp_str, level, message = match.groups()
                    timestamp = datetime.strptime(
                        timestamp_str,
                        '%Y-%m-%d %H:%M:%S'
                    )
                    
                    yield {
                        'timestamp': timestamp,
                        'level': level,
                        'message': message
                    }
    
    def filter_by_level(self, level: str) -> Generator[Dict, None, None]:
        """Filter logs by level."""
        return (
            entry for entry in self.parse_log_lines()
            if entry['level'].upper() == level.upper()
        )
    
    def group_by_hour(self) -> Dict[str, List[Dict]]:
        """Group logs by hour."""
        groups = defaultdict(list)
        for entry in self.parse_log_lines():
            hour = entry['timestamp'].strftime('%Y-%m-%d %H:00')
            groups[hour].append(entry)
        return dict(groups)

# Example usage
def demonstrate_log_analysis():
    # Create sample log file
    log_content = """
2024-04-15 09:00:01 [INFO] Application started
2024-04-15 09:00:05 [DEBUG] Connecting to database
2024-04-15 09:00:06 [INFO] Database connected
2024-04-15 09:01:23 [WARNING] High memory usage
2024-04-15 09:02:45 [ERROR] Database query failed
2024-04-15 09:02:50 [INFO] Retrying connection
2024-04-15 09:03:01 [INFO] Connection restored
2024-04-15 10:00:00 [INFO] Starting backup
""".strip()
    
    with open('application.log', 'w') as f:
        f.write(log_content)
    
    # Create analyzer
    analyzer = LogAnalyzer('application.log')
    
    # Process logs with generator
    print("All log entries:")
    for entry in analyzer.parse_log_lines():
        print(f"{entry['timestamp']} [{entry['level']}] "
              f"{entry['message']}")
    
    # Filter ERROR logs
    print("\nERROR logs:")
    for entry in analyzer.filter_by_level('ERROR'):
        print(f"{entry['timestamp']} - {entry['message']}")
    
    # Group by hour
    print("\nLogs by hour:")
    hourly_logs = analyzer.group_by_hour()
    for hour, entries in hourly_logs.items():
        print(f"\n{hour}:")
        for entry in entries:
            print(f"  [{entry['level']}] {entry['message']}")

# Run demonstration
demonstrate_log_analysis()
```

## 4. Functional Programming Tools

### Map, Filter, and Reduce
```python
from functools import reduce
from typing import List, Dict, Any
from datetime import datetime

class DataAnalyzer:
    @staticmethod
    def transform_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform data using map."""
        return list(map(
            lambda x: {
                **x,
                'processed_at': datetime.now().isoformat(),
                'value_with_tax': x['value'] * 1.2
            },
            data
        ))
    
    @staticmethod
    def filter_valid_records(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter valid records."""
        return list(filter(
            lambda x: x['value'] > 0 and 'id' in x,
            data
        ))
    
    @staticmethod
    def calculate_total(data: List[Dict[str, Any]]) -> float:
        """Calculate total using reduce."""
        return reduce(
            lambda acc, x: acc + x['value'],
            data,
            0
        )

# Example usage
def demonstrate_functional_tools():
    # Sample data
    data = [
        {'id': 1, 'value': 100},
        {'id': 2, 'value': 200},
        {'value': 0},  # Invalid record
        {'id': 3, 'value': 300},
    ]
    
    analyzer = DataAnalyzer()
    
    # Transform data
    transformed = analyzer.transform_data(data)
    print("Transformed data:")
    print(json.dumps(transformed, indent=2))
    
    # Filter valid records
    valid_records = analyzer.filter_valid_records(data)
    print("\nValid records:")
    print(json.dumps(valid_records, indent=2))
    
    # Calculate total
    total = analyzer.calculate_total(valid_records)
    print(f"\nTotal value: ${total:,.2f}")

# Run demonstration
demonstrate_functional_tools()
```

## 5. Best Practices

### Lambda Functions
1. Use for simple operations
2. Keep them readable
3. Consider named functions for complex operations
4. Perfect for key functions in sorting/filtering

### Decorators
1. Use `@wraps` to preserve function metadata
2. Keep them focused on one responsibility
3. Use for cross-cutting concerns
4. Document the behavior modification

### Generators
1. Use for large datasets
2. Implement `__iter__` for classes
3. Use generator expressions for simple cases
4. Consider memory usage

## 6. Knowledge Check ✅

1. What are the key differences between lambda functions and regular functions?
2. How do decorators modify function behavior and when should you use them?
3. What are the benefits of using generators over lists?
4. How do map, filter, and reduce functions work together?
5. What makes a function "pure" and why is it important?
6. When should you use list comprehensions vs map/filter?
7. How do you handle errors in functional programming?
8. What are the performance implications of functional programming?

## 7. Summary

### Key Takeaways
- Use lambda functions for simple, one-line operations
- Apply decorators to modify function behavior cleanly
- Implement generators for memory-efficient iteration
- Choose appropriate functional tools (map, filter, reduce)
- Write pure functions for better maintainability

## 📚 Additional Resources
- [Python Functional Programming HOWTO](https://docs.python.org/3/howto/functional.html)
- [Real Python - Functional Programming](https://realpython.com/python-functional-programming/)
- [Python Decorators Guide](https://realpython.com/primer-on-python-decorators/)

---

> **Navigation**
> - [← Python File Handling](07-Python-File-Handling.md)
> - [Python Project Setup →](09-Python-Project-Setup.md)
