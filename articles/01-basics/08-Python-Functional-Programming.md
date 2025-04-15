# Day - 8 Functional Programming in Python

## Overview
This lesson explores functional programming concepts in Python, including lambda functions, decorators, and generators. You'll learn how to write more concise and efficient code using these powerful features, with practical examples from real-world scenarios.

## Learning Objectives
- Master lambda functions for concise operations
- Understand and create decorators
- Implement generators for memory efficiency
- Apply functional programming concepts
- Use built-in functional tools (map, filter, reduce)

## Prerequisites
- Completion of [01 - Python Basics](01-Python-Basics-Variables-Types-Operators.md)
- Completion of [02 - Flow Control](02-Python-Flow-Control-Loops-Conditions.md)
- Completion of [03 - Functions](03-Python-Functions-Modular-Programming.md)
- Completion of [04 - Modules and Packages](04-Python-Modules-Packages.md)
- Completion of [05 - Object-Oriented Programming](05-Python-OOP.md)
- Completion of [06 - File Handling](06-Python-File-Handling.md)
- Completion of [07 - Testing and Debugging](07-Python-Testing-Debugging.md)
- Python 3.x installed on your computer

## Time Estimate
- Reading: 40 minutes
- Practice: 50 minutes
- Assignments: 45 minutes

---

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

## Summary

### Key Takeaways
1. Lambda functions for quick, simple operations
2. Decorators for modifying function behavior
3. Generators for memory-efficient iteration
4. Functional programming for cleaner code

### What's Next
- [Testing and Debugging](08-Python-Testing-Debugging.md)
- Working with Databases
- Web Development

---

> **Navigation**
> - [← File Handling](06-Python-File-Handling.md)
> - [Testing and Debugging →](08-Python-Testing-Debugging.md)
