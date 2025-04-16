# Day 3 - Functions and Modular Programming: Building Reusable Code

## Overview
In programming, functions are the building blocks that help us organize code into reusable pieces. Think of functions like recipes in a cookbook - they take ingredients (inputs), follow a set of instructions, and produce a dish (output). This lesson will teach you how to create and use functions effectively in Python, making your code more organized, reusable, and maintainable.

### Real-World Scenarios
1. **Calculator App**: Functions for each operation (add, subtract, multiply, divide)
2. **Data Processing**: Functions to clean, transform, and analyze data
3. **Game Development**: Functions for player movement, scoring, and game logic
4. **Web Applications**: Functions to handle user input, validate data, and process forms

## Learning Objectives
- Create and use functions effectively with modern Python syntax
- Understand parameter types, default values, and return types
- Master function scope and variable visibility
- Apply modular programming principles
- Implement proper error handling and validation

## Prerequisites
- Understanding of Python variables and data types
- Knowledge of flow control (if statements, loops)
- Python 3.11+ installed

## Time Estimate
- Reading: 30 minutes
- Practice: 45 minutes
- Assignments: 40 minutes

---

## 1. Function Fundamentals

### What is a Function?
A function is a named block of organized, reusable code that performs a specific task. Functions help us:
- Avoid repeating code (DRY - Don't Repeat Yourself)
- Break down complex problems into smaller, manageable pieces
- Make code more readable and maintainable
- Enable code reuse across different parts of a program

### 1.1 Basic Function Structure
```python
def calculate_area(width: float, height: float) -> float:
    """Calculate the area of a rectangle.
    
    Args:
        width (float): The width of the rectangle
        height (float): The height of the rectangle
        
    Returns:
        float: The calculated area
    """
    return width * height

# Using the function
area = calculate_area(5.0, 3.0)
print(f"Area: {area} square units")
```

Let's break down the function structure:
1. **def** keyword: Tells Python we're defining a function
2. **Function name**: Should be descriptive and follow Python naming conventions
3. **Parameters**: Input values the function expects (with type hints)
4. **Return type**: What the function will output (-> float)
5. **Docstring**: Documentation explaining what the function does
6. **Function body**: The actual code that performs the task
7. **return statement**: Specifies what value to send back

### 1.2 Understanding Function Parameters
Python offers several ways to pass data to functions through parameters. Understanding these options helps you design more flexible and maintainable functions.

#### Types of Parameters:
1. **Positional Parameters**: Regular parameters that must be provided in order
2. **Keyword Parameters**: Parameters that can be specified by name
3. **Default Parameters**: Parameters with predefined values
4. **Keyword-Only Parameters**: Must be specified using their names
5. **Variable Positional (*args)**: Accept any number of positional arguments
6. **Variable Keyword (**kwargs)**: Accept any number of keyword arguments

```python
# Example: User Profile System

def create_user(
    # Required positional parameters
    name: str,
    age: int,
    # Keyword-only parameters (note the *)
    *,
    email: str,
    # Optional parameters with defaults
    active: bool = True,
    role: str = "user",
    # Variable keyword parameters
    **profile_data: Any
) -> dict:
    """Create a user profile with flexible attributes.
    
    Args:
        name (str): User's full name
        age (int): User's age in years
        email (str): User's email address (keyword-only)
        active (bool, optional): Account status. Defaults to True.
        role (str, optional): User role. Defaults to "user".
        **profile_data: Additional profile information
        
    Returns:
        dict: Complete user profile
        
    Example:
        >>> user = create_user(
        ...     "John Doe",
        ...     25,
        ...     email="john@example.com",
        ...     location="New York",
        ...     interests=["Python", "AI"]
        ... )
    """
    # Start with base profile
    profile = {
        'name': name.title(),
        'age': age,
        'email': email.lower(),
        'active': active,
        'role': role
    }
    
    # Add any additional profile data
    profile.update(profile_data)
    
    return profile

# Example Usage:

# 1. Basic usage with required parameters
user1 = create_user(
    "john doe",
    25,
    email="john@example.com"
)

# 2. Using optional parameters
user2 = create_user(
    "jane smith",
    30,
    email="jane@example.com",
    active=False,
    role="admin"
)

# 3. Adding custom profile data
user3 = create_user(
    "bob wilson",
    35,
    email="bob@example.com",
    location="London",
    department="Engineering",
    skills=["Python", "JavaScript", "SQL"]
)

# Function with variable arguments
def calculate_stats(*numbers: float, precision: int = 2) -> dict:
    """Calculate statistics for any number of values.
    
    Args:
        *numbers: Variable number of numeric values
        precision: Decimal places for results (default: 2)
        
    Returns:
        dict: Statistical measures
    """
    if not numbers:
        raise ValueError("At least one number is required")
        
    return {
        'count': len(numbers),
        'sum': round(sum(numbers), precision),
        'average': round(sum(numbers) / len(numbers), precision),
        'min': round(min(numbers), precision),
        'max': round(max(numbers), precision)
    }

# Using variable arguments
print(calculate_stats(1.234, 5.678, 9.101, precision=3))
print(calculate_stats(*[2.5, 3.7, 4.9]))
```

#### Parameter Design Best Practices:
1. **Order Parameters Logically:**
   - Required parameters first
   - Optional parameters with defaults later
   - Variable arguments last

2. **Use Keyword-Only Arguments:**
   - For parameters that should be explicitly named
   - Improves code readability
   - Prevents parameter order confusion

3. **Choose Good Default Values:**
   - Make defaults immutable (strings, numbers, None, etc.)
   - Avoid mutable defaults (lists, dicts, etc.)
   - Use None and create mutable objects inside the function

4. **Document Parameters Clearly:**
   - Use type hints for better IDE support
   - Write clear docstrings with examples
   - Explain parameter relationships if any

### 1.3 Type Hints and Annotations
```python
from typing import List, Dict, Optional, Union

def process_items(items: List[int],
                 multiplier: Optional[float] = None,
                 settings: Dict[str, Union[str, int]] = None) -> List[float]:
    """Process a list of items with optional multiplier and settings.
    
    Args:
        items: List of integers to process
        multiplier: Optional scaling factor
        settings: Configuration dictionary
        
    Returns:
        List of processed values
    """
    if multiplier is None:
        multiplier = 1.0
    
    settings = settings or {}
    base_value = float(settings.get('base_value', 0))
    
    return [base_value + (item * multiplier) for item in items]

# Example usage
data = [1, 2, 3, 4, 5]
config = {'base_value': 10}

result1 = process_items(data)  # Default multiplier
result2 = process_items(data, 2.5, config)  # With multiplier and settings
```
## 2. Advanced Function Features

### 2.1 Function Decorators
Decorators are a powerful way to modify or enhance functions without changing their code directly. They follow the Decorator pattern and use the `@` syntax in Python. Think of decorators as wrappers that add extra functionality to existing functions.

**Common Uses of Decorators:**
1. Logging and debugging
2. Timing and performance monitoring
3. Access control and authentication
4. Caching and memoization
5. Input validation
6. Error handling

```python
from functools import wraps
from time import time
from typing import Callable, Any

# Example 1: Timing Decorator
def timing_decorator(func: Callable) -> Callable:
    """Measure execution time of a function."""
    @wraps(func)  # Preserves the original function's metadata
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

# Example 2: Validation Decorator
def validate_positive(func: Callable) -> Callable:
    """Ensure all numeric arguments are positive."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if any(isinstance(arg, (int, float)) and arg <= 0 for arg in args):
            raise ValueError("Arguments must be positive numbers")
        return func(*args, **kwargs)
    return wrapper

# Example 3: Caching Decorator
def memoize(func: Callable) -> Callable:
    """Cache function results for better performance."""
    cache = {}
    @wraps(func)
    def wrapper(*args: Any) -> Any:
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

# Using multiple decorators
@timing_decorator
@validate_positive
@memoize
def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

# The decorators will:
# 1. Cache results for repeated calls
# 2. Validate that n is positive
# 3. Print execution time
result = calculate_fibonacci(10)

# Example 4: Decorator with Arguments
def repeat(times: int) -> Callable:
    """Repeat a function a specified number of times."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)
def greet(name: str) -> None:
    print(f"Hello, {name}!")

# Will print greeting 3 times
greet("Alice")
```

### 2.2 Lambda Functions
Lambda functions, also known as anonymous functions, are small, one-time-use functions that can have any number of arguments but only one expression. They are perfect for simple operations that you don't need to define as full functions.

**When to use lambda functions:**
- As arguments to higher-order functions (map, filter, sorted)
- For simple operations that don't need a full function definition
- In data processing pipelines where you need quick transformations

```python
# Sort complex data
users = [
    {'name': 'Alice', 'age': 30},
    {'name': 'Bob', 'age': 25},
    {'name': 'Charlie', 'age': 35}
]

# Regular function vs Lambda function
def get_age(user):
    return user['age']

# These two lines do the same thing:
sorted_by_age_1 = sorted(users, key=get_age)                # Using regular function
sorted_by_age_2 = sorted(users, key=lambda x: x['age'])    # Using lambda

# Common lambda use cases:
# 1. Filtering data
adults = filter(lambda x: x['age'] >= 18, users)

# 2. Transforming data
names = map(lambda x: x['name'].upper(), users)

# 3. Custom sorting
users.sort(key=lambda x: (x['age'], x['name']))  # Sort by age, then name
```

### 2.3 Generator Functions
Generator functions are a powerful way to create iterators that generate values on-the-fly, saving memory when working with large sequences. They use the `yield` keyword to produce a series of values over time, rather than computing them all at once.

**Benefits of Generators:**
- Memory efficient: Values are generated one at a time
- Perfect for large datasets
- Can represent infinite sequences
- Lazy evaluation: Values are computed only when needed

```python
from typing import Generator, Iterator

def number_generator(start: int, end: int) -> Generator[int, None, None]:
    """Generate a sequence of numbers.
    
    This is memory efficient for large ranges because it generates
    numbers one at a time instead of storing them all in memory.
    """
    current = start
    while current < end:
        yield current  # Pause here and return the current value
        current += 1

def fibonacci_sequence(limit: int) -> Iterator[int]:
    """Generate Fibonacci sequence up to limit.
    
    This is a perfect use case for generators because each number
    depends on the previous ones, but we don't need to store them all.
    """
    a, b = 0, 1
    while a < limit:
        yield a
        a, b = b, a + b

# Example 1: Memory-efficient processing of large sequences
for num in number_generator(1, 1000000):
    if num % 100000 == 0:
        print(f"Processed {num} numbers")

# Example 2: Generate first 10 Fibonacci numbers
fibs = list(fibonacci_sequence(100))[:10]
print(f"First 10 Fibonacci numbers: {fibs}")

# Example 3: Pipeline of generators
def even_numbers(numbers):
    for n in numbers:
        if n % 2 == 0:
            yield n

def square_numbers(numbers):
    for n in numbers:
        yield n * n

# Create a pipeline: generate numbers -> filter evens -> square them
numbers = number_generator(1, 10)           # 1, 2, 3, 4, 5, 6, 7, 8, 9
evens = even_numbers(numbers)               # 2, 4, 6, 8
squares = square_numbers(evens)             # 4, 16, 36, 64

print("Squares of even numbers:")
for square in squares:
    print(square)
```

### 2.4 Function Overloading with Singledispatch
Function overloading allows a function to handle different types of inputs differently. While Python doesn't support traditional overloading like Java or C++, it provides the `singledispatch` decorator for implementing similar functionality based on argument types.

**Benefits of Function Overloading:**
1. Type-specific behavior without complex if-else chains
2. Clean, maintainable code for handling different data types
3. Easy to add support for new types
4. Better type safety and error messages

```python
from functools import singledispatch
from typing import Union, List, Dict, Any
from decimal import Decimal

# Example: Data Processing System

@singledispatch
def process_data(data: Any) -> str:
    """Process different types of data.
    
    This is the base implementation that handles unknown types.
    Each registered handler will override this for specific types.
    """
    raise TypeError(f"Unsupported type: {type(data)}")

@process_data.register
def _(data: int) -> str:
    """Handle integer data."""
    # Convert to binary and hex
    return f"Int: {data} (bin: {bin(data)}, hex: {hex(data)})"

@process_data.register
def _(data: float) -> str:
    """Handle floating-point data."""
    # Format with different precisions
    return f"Float: {data:.2f} (scientific: {data:.2e})"

@process_data.register
def _(data: Decimal) -> str:
    """Handle decimal data."""
    # Show as percentage if < 1, otherwise normal
    if data < 1:
        return f"Decimal: {data:.1%}"
    return f"Decimal: {data:.2f}"

@process_data.register
def _(data: str) -> str:
    """Handle string data."""
    # Basic string analysis
    return f"String: {data.upper()} (len: {len(data)}, words: {len(data.split())})"

@process_data.register
def _(data: List[Union[int, str]]) -> str:
    """Handle list data."""
    # Analyze list contents
    types = [type(x).__name__ for x in data]
    return f"List: {len(data)} items, types: {set(types)}"

@process_data.register
def _(data: Dict[str, Any]) -> str:
    """Handle dictionary data."""
    # Summarize dict structure
    return f"Dict: {len(data)} keys: {', '.join(data.keys())}"

# Example usage with different types
def demonstrate_processing():
    test_data = [
        42,                     # Integer
        3.14159,                # Float
        Decimal('0.75'),        # Decimal
        "Hello, World!",        # String
        [1, 2, "three"],        # Mixed List
        {"a": 1, "b": "two"}    # Dictionary
    ]
    
    print("Data Processing Examples:")
    for data in test_data:
        try:
            result = process_data(data)
            print(f"✓ {result}")
        except TypeError as e:
            print(f"✗ Error: {e}")

if __name__ == '__main__':
    demonstrate_processing()
```

**Key Points about Singledispatch:**
1. The base function handles the default case
2. Each registered function handles a specific type
3. Registration is done using the `@function.register` decorator
4. Python chooses the most specific handler for each type
5. New types can be added without modifying existing code

## 3. Best Practices and Common Pitfalls

### 3.1 Function Design Principles
```python
# ✅ DO: Single Responsibility Principle
def calculate_total_price(items: List[Dict[str, Union[str, float]]]) -> float:
    """Calculate total price of items with tax."""
    return sum(item['price'] for item in items)

def apply_tax(total: float, tax_rate: float = 0.1) -> float:
    """Apply tax to total amount."""
    return total * (1 + tax_rate)

# ❌ DON'T: Mix multiple responsibilities
def process_order(items, tax_rate, shipping, payment):
    # Too many responsibilities in one function
    # Calculating total, tax, shipping, and handling payment
    pass

# ✅ DO: Clear parameter names and type hints
def create_account(
    username: str,
    email: str,
    *,  # Force keyword arguments
    password: str,
    is_admin: bool = False
) -> Dict[str, Any]:
    """Create a new user account."""
    return {
        'username': username,
        'email': email,
        'password': hash_password(password),
        'is_admin': is_admin
    }

# ❌ DON'T: Use unclear parameter names
def create_acc(u, e, p, a=False):  # Unclear parameter names
    pass
```

### 3.2 Common Pitfalls
```python
# ❌ DON'T: Use mutable default arguments
def add_item(item: Dict, items: List = []):  # Bad: mutable default
    items.append(item)
    return items

# ✅ DO: Use None as default
def add_item(item: Dict, items: Optional[List] = None) -> List:
    if items is None:
        items = []
    items.append(item)
    return items

# ❌ DON'T: Return different types
def get_user(user_id: int) -> Union[Dict, None, str]:
    if not user_id:
        return None
    if user_id < 0:
        return "Invalid ID"  # Inconsistent return type
    return {'id': user_id}

# ✅ DO: Consistent return types with proper error handling
from dataclasses import dataclass
from typing import Optional

@dataclass
class User:
    id: int
    name: str

def get_user(user_id: int) -> Optional[User]:
    if not user_id or user_id < 0:
        return None
    return User(id=user_id, name="John Doe")
```

## 4. Practice Exercises

### 🎯 Exercise 1: Task Management System
Create a task management system using functions. This exercise will help you practice:
- Function parameters and return types
- Working with different data types
- Using decorators for logging
- Error handling

```python
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum, auto
import json

# Step 1: Define data structures
class Priority(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()

@dataclass
class Task:
    title: str
    description: str
    priority: Priority
    due_date: datetime
    completed: bool = False
    tags: List[str] = None

# Step 2: Create the task manager
class TaskManager:
    def __init__(self):
        self.tasks: List[Task] = []
    
    def add_task(
        self,
        title: str,
        description: str,
        priority: Priority,
        due_days: int,
        tags: Optional[List[str]] = None
    ) -> Task:
        """Add a new task to the system."""
        task = Task(
            title=title,
            description=description,
            priority=priority,
            due_date=datetime.now() + timedelta(days=due_days),
            tags=tags or []
        )
        self.tasks.append(task)
        return task
    
    def get_tasks_by_priority(self, priority: Priority) -> List[Task]:
        """Get all tasks with specified priority."""
        return [task for task in self.tasks if task.priority == priority]
    
    def get_overdue_tasks(self) -> List[Task]:
        """Get all overdue tasks."""
        now = datetime.now()
        return [task for task in self.tasks 
                if not task.completed and task.due_date < now]
    
    def complete_task(self, title: str) -> bool:
        """Mark a task as completed."""
        for task in self.tasks:
            if task.title == title:
                task.completed = True
                return True
        return False

# Step 3: Add logging decorator
def log_action(func):
    """Log all task management actions."""
    def wrapper(*args, **kwargs):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = func(*args, **kwargs)
        print(f"[{timestamp}] Called {func.__name__} with args: {args[1:]} {kwargs}")
        return result
    return wrapper

# Step 4: Create task report generator
def generate_report(manager: TaskManager) -> Dict:
    """Generate a summary report of all tasks."""
    total_tasks = len(manager.tasks)
    completed = sum(1 for task in manager.tasks if task.completed)
    overdue = len(manager.get_overdue_tasks())
    
    by_priority = {
        priority.name: len(manager.get_tasks_by_priority(priority))
        for priority in Priority
    }
    
    return {
        'total_tasks': total_tasks,
        'completed': completed,
        'overdue': overdue,
        'by_priority': by_priority,
        'completion_rate': f"{(completed/total_tasks)*100:.1f}%" if total_tasks else "0%"
    }

# Example usage and testing
def test_task_system():
    # Create manager
    manager = TaskManager()
    
    # Add some tasks
    manager.add_task(
        "Complete project",
        "Finish the Python project",
        Priority.HIGH,
        due_days=2,
        tags=["python", "coding"]
    )
    
    manager.add_task(
        "Review code",
        "Code review for team",
        Priority.MEDIUM,
        due_days=1,
        tags=["review"]
    )
    
    manager.add_task(
        "Update docs",
        "Update documentation",
        Priority.LOW,
        due_days=5,
        tags=["docs"]
    )
    
    # Complete a task
    manager.complete_task("Review code")
    
    # Generate and print report
    report = generate_report(manager)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    test_task_system()
```

### 🎯 Exercise 2: Data Analysis Pipeline
Create a data analysis pipeline using functional programming concepts. Practice:
- Function composition
- Generator functions
- Type hints
- Data transformation

```python
from typing import List, Iterator, Dict, Any
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime
import statistics

@dataclass
class SalesRecord:
    date: datetime
    product_id: str
    quantity: int
    price: Decimal
    customer_id: str

# Step 1: Data Loading
def load_sales_data(filename: str) -> Iterator[SalesRecord]:
    """Load sales records from a file."""
    # Simulate loading data
    sample_data = [
        ("2023-01-01", "P1", 5, "10.99", "C1"),
        ("2023-01-01", "P2", 2, "25.50", "C2"),
        ("2023-01-02", "P1", 3, "10.99", "C3"),
    ]
    
    for date_str, pid, qty, price, cid in sample_data:
        yield SalesRecord(
            date=datetime.strptime(date_str, "%Y-%m-%d"),
            product_id=pid,
            quantity=qty,
            price=Decimal(price),
            customer_id=cid
        )

# Step 2: Data Transformation
def calculate_total(record: SalesRecord) -> Decimal:
    """Calculate total sale amount."""
    return record.price * record.quantity

def group_by_product(records: List[SalesRecord]) -> Dict[str, List[SalesRecord]]:
    """Group records by product ID."""
    result = {}
    for record in records:
        if record.product_id not in result:
            result[record.product_id] = []
        result[record.product_id].append(record)
    return result

# Step 3: Analysis Functions
def analyze_product_sales(records: List[SalesRecord]) -> Dict[str, Any]:
    """Analyze sales for a product."""
    if not records:
        return {}
        
    totals = [calculate_total(r) for r in records]
    quantities = [r.quantity for r in records]
    
    return {
        'total_revenue': sum(totals),
        'average_order': statistics.mean(quantities),
        'order_count': len(records),
        'total_units': sum(quantities)
    }

# Step 4: Report Generation
def generate_sales_report(filename: str) -> Dict[str, Any]:
    """Generate complete sales report."""
    # Load and process data
    records = list(load_sales_data(filename))
    by_product = group_by_product(records)
    
    # Analyze each product
    product_analysis = {
        pid: analyze_product_sales(precs)
        for pid, precs in by_product.items()
    }
    
    # Calculate totals
    total_revenue = sum(pa['total_revenue'] 
                       for pa in product_analysis.values())
    total_orders = sum(pa['order_count'] 
                      for pa in product_analysis.values())
    
    return {
        'total_revenue': total_revenue,
        'total_orders': total_orders,
        'products': product_analysis
    }

# Example usage
def test_analysis_pipeline():
    report = generate_sales_report('sales.csv')
    
    print("Sales Analysis Report")
    print("-" * 20)
    print(f"Total Revenue: ${report['total_revenue']}")
    print(f"Total Orders: {report['total_orders']}")
    print("\nProduct Details:")
    
    for pid, analysis in report['products'].items():
        print(f"\nProduct {pid}:")
        print(f"  Revenue: ${analysis['total_revenue']}")
        print(f"  Average Order: {analysis['average_order']:.1f} units")
        print(f"  Total Units: {analysis['total_units']}")

if __name__ == '__main__':
    test_analysis_pipeline()
```

### 🎯 Exercise 3: Challenge
Extend either the Task Manager or Sales Analysis system with these features:
1. Add error handling with custom exceptions
2. Implement data validation using decorators
3. Add data persistence (save/load from file)
4. Create a command-line interface
5. Add unit tests for your functions

## 📚 Additional Resources
- [Python Functions Documentation](https://docs.python.org/3/tutorial/controlflow.html#defining-functions)
- [PEP 484 – Type Hints](https://peps.python.org/pep-0484/)
- [Real Python: Decorators](https://realpython.com/primer-on-python-decorators/)
- [Python Type Checking](https://realpython.com/python-type-checking/)

## ✅ Knowledge Check
1. What are the benefits of using type hints in function definitions?
2. How do generator functions differ from regular functions?
3. When should you use lambda functions vs regular functions?
4. What is function overloading and when is it useful?
5. Why should you avoid mutable default arguments?

## 🔍 Common Issues and Solutions
| Issue | Solution |
|-------|----------|
| Mutable Defaults | Use None and create new object in function |
| Type Consistency | Use Union/Optional for multiple return types |
| Long Functions | Break into smaller, focused functions |
| Side Effects | Make functions pure when possible |

## 📝 Summary
- Functions are building blocks for modular code
- Modern Python features enhance function clarity and safety
- Type hints provide better documentation and IDE support
- Best practices focus on clarity and maintainability
- Advanced features like decorators and generators add power

> **Navigation**
> - [← Flow Control](02-Python-Flow-Control-Loops-Conditions.md)
> - [Modules and Packages →](04-Python-Modules-Packages.md)
```

### Keyword Arguments
```python
def display_info(name: str, age: int, city: str) -> None:
    """Display person information."""
    print(f"Name: {name}")
    print(f"Age: {age}")
    print(f"City: {city}")

# Using keyword arguments
display_info(age=30, city="New York", name="Alice")
```

## 2. Advanced Function Concepts

### Variable Scope and Lifetime
```python
# Global vs Local variables
global_var = 10

def demonstrate_scope():
    local_var = 20  # Local variable
    print(f"Inside function - Local: {local_var}, Global: {global_var}")

demonstrate_scope()
print(f"Outside function - Global: {global_var}")
# print(local_var)  # This would cause an error - local_var is not accessible
```

### Args and Kwargs
```python
# Variable number of arguments
def calculate_sum(*args):
    """Calculate sum of any number of arguments"""
    return sum(args)

# Variable number of keyword arguments
def create_user(**kwargs):
    """Create user profile from keyword arguments"""
    print("User Profile:")
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# Test the functions
print(calculate_sum(1, 2, 3, 4, 5))  # Output: 15
create_user(name="John", age=30, occupation="Developer")
```

### Lambda Functions
```python
# Simple lambda function
square = lambda x: x**2

# Using lambda with map
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))

# Using lambda with filter
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))

print(f"Original numbers: {numbers}")
print(f"Squared: {squared}")
print(f"Even numbers: {even_numbers}")
```

### Function Decorators
```python
# Simple decorator to measure execution time
import time

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds to execute")
        return result
    return wrapper

# Using the decorator
@measure_time
def slow_function():
    time.sleep(1)  # Simulate a slow operation
    return "Done!"

# Test the decorated function
result = slow_function()
```

---

## 3. Modular Programming and Real-world Applications

### Banking System Example
```python
class BankAccount:
    def __init__(self, account_number, balance=0):
        self.account_number = account_number
        self.balance = balance
        self.transactions = []

    def deposit(self, amount):
        """Deposit money into account"""
        if amount > 0:
            self.balance += amount
            self.transactions.append(("deposit", amount))
            return f"Deposited ${amount:.2f}. New balance: ${self.balance:.2f}"
        return "Invalid deposit amount"

    def withdraw(self, amount):
        """Withdraw money from account"""
        if 0 < amount <= self.balance:
            self.balance -= amount
            self.transactions.append(("withdraw", amount))
            return f"Withdrew ${amount:.2f}. New balance: ${self.balance:.2f}"
        return "Insufficient funds or invalid amount"

    def get_statement(self):
        """Generate account statement"""
        statement = f"Account Statement for {self.account_number}\n"
        statement += "-" * 40 + "\n"
        for transaction_type, amount in self.transactions:
            statement += f"{transaction_type.title()}: ${amount:.2f}\n"
        statement += "-" * 40 + "\n"
        statement += f"Current Balance: ${self.balance:.2f}"
        return statement

# Test the banking system
def test_banking_system():
    # Create a new account
    account = BankAccount("1234567890")
    
    # Perform transactions
    print(account.deposit(1000))
    print(account.withdraw(500))
    print(account.deposit(250))
    print(account.withdraw(1000))  # Should fail
    
    # Print statement
    print("\n" + account.get_statement())

# Run the test
test_banking_system()
```

### File Processing System
```python
import os
from typing import List, Dict

class FileProcessor:
    def __init__(self, directory: str):
        self.directory = directory

    def get_files_by_extension(self, extension: str) -> List[str]:
        """Get all files with specific extension"""
        files = []
        for file in os.listdir(self.directory):
            if file.endswith(extension):
                files.append(os.path.join(self.directory, file))
        return files

    def analyze_file(self, filepath: str) -> Dict:
        """Analyze a text file and return statistics"""
        try:
            with open(filepath, 'r') as file:
                content = file.read()
                lines = content.split('\n')
                words = content.split()
                characters = len(content)
                return {
                    'lines': len(lines),
                    'words': len(words),
                    'characters': characters,
                    'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0
                }
        except Exception as e:
            return {'error': str(e)}

    def process_directory(self, extension: str = '.txt') -> List[Dict]:
        """Process all files with given extension in directory"""
        results = []
        files = self.get_files_by_extension(extension)
        
        for file in files:
            stats = self.analyze_file(file)
            stats['filename'] = os.path.basename(file)
            results.append(stats)
        
        return results

# Example usage
def test_file_processor():
    processor = FileProcessor('./documents')
    results = processor.process_directory('.txt')
    
    print("File Analysis Results:")
    for result in results:
        print(f"\nFile: {result['filename']}")
        for key, value in result.items():
            if key != 'filename':
                print(f"{key.replace('_', ' ').title()}: {value}")

# Note: This requires a directory with text files to test
# test_file_processor()
```

---

## 1. Introduction to Functions

### What is a Function?
- A function is a reusable block of code that performs a specific task
- It helps avoid code duplication and makes programs more organized
- Functions can take inputs (parameters) and return outputs

### Basic Function Syntax
```python
def function_name(parameter1, parameter2, ...):
    """Docstring explaining what the function does"""
    # Function body
    # Code to be executed
    return result  # Optional return statement
```

### Simple Function Examples
```python
# Function without parameters
def greet():
    """Print a simple greeting"""
    print("Hello, World!")

# Function with parameters
def personalized_greet(name):
    """Print a personalized greeting"""
    print(f"Hello, {name}!")

# Function with return value
def add_numbers(a, b):
    """Add two numbers and return the result"""
    return a + b

# Test the functions
greet()  # Output: Hello, World!
personalized_greet("Alice")  # Output: Hello, Alice!
result = add_numbers(5, 3)
print(result)  # Output: 8
```

### Function Parameters
```python
# Default parameters
def greet_with_title(name, title="Mr."):
    print(f"Hello, {title} {name}")

# Keyword arguments
def display_info(name, age, city):
    print(f"{name} is {age} years old and lives in {city}")

# Test the functions
greet_with_title("Smith")  # Uses default title
greet_with_title("Johnson", "Dr.")  # Overrides default title

# Using keyword arguments (order doesn't matter)
display_info(age=30, city="New York", name="Alice")
```

## 4. Best Practices and Code Standards

### Function Design Principles

#### ✅ DO: Write Single-Purpose Functions
```python
# Bad - function does multiple things
def process_user_data(user_data):
    validate_data(user_data)  # Should be separate
    save_to_database(user_data)  # Should be separate
    send_email(user_data)  # Should be separate

# Good - single responsibility
def validate_user_data(user_data):
    return all(required_fields_present(user_data))

def save_user_data(user_data):
    return database.save(user_data)

def send_welcome_email(user_email):
    return mailer.send(user_email, template='welcome')
```

#### ✅ DO: Use Clear and Descriptive Names
```python
# Bad
def calc(x, y):
    return x * y

# Good
def calculate_rectangle_area(length, width):
    return length * width
```

#### ✅ DO: Include Docstrings and Type Hints
```python
from typing import List, Dict

def process_user_records(records: List[Dict]) -> Dict:
    """Process user records and return summary statistics.
    
    Args:
        records: List of dictionaries containing user data
        
    Returns:
        Dictionary containing summary statistics
    """
    # Function implementation
```

#### ❌ DON'T: Use Global Variables
```python
# Bad - using global variable
total = 0

def add_to_total(value):
    global total
    total += value

# Good - pass and return values
def add_to_total(current_total, value):
    return current_total + value
```

### Error Handling Best Practices

#### ✅ DO: Use Try-Except Blocks
```python
def safe_divide(a: float, b: float) -> float:
    """Safely divide two numbers."""
    try:
        return a / b
    except ZeroDivisionError:
        print("Error: Division by zero!")
        return 0
    except TypeError:
        print("Error: Invalid number type!")
        return 0
```

#### ✅ DO: Raise Custom Exceptions
```python
class InvalidAgeError(Exception):
    pass

def validate_age(age: int) -> bool:
    """Validate user age."""
    if not isinstance(age, int):
        raise TypeError("Age must be an integer")
    if age < 0:
        raise InvalidAgeError("Age cannot be negative")
    if age > 150:
        raise InvalidAgeError("Age is unreasonably high")
    return True
```

## 5. Hands-on Practice

### Exercise 1: Contact Manager (Basic)
```python
class Contact:
    def __init__(self, name: str, email: str, phone: str):
        self.name = name
        self.email = email
        self.phone = phone

class ContactManager:
    def __init__(self):
        self.contacts = []
    
    def add_contact(self, name: str, email: str, phone: str) -> None:
        """Add a new contact."""
        contact = Contact(name, email, phone)
        self.contacts.append(contact)
        print(f"Added contact: {name}")
    
    def find_contact(self, name: str) -> Contact:
        """Find a contact by name."""
        for contact in self.contacts:
            if contact.name.lower() == name.lower():
                return contact
        return None
    
    def display_contacts(self) -> None:
        """Display all contacts."""
        if not self.contacts:
            print("No contacts found.")
            return
        
        print("\nContact List:")
        print("-" * 40)
        for contact in self.contacts:
            print(f"Name: {contact.name}")
            print(f"Email: {contact.email}")
            print(f"Phone: {contact.phone}")
            print("-" * 40)

# Test the contact manager
def test_contact_manager():
    manager = ContactManager()
    
    # Add some contacts
    manager.add_contact("John Doe", "john@example.com", "123-456-7890")
    manager.add_contact("Jane Smith", "jane@example.com", "098-765-4321")
    
    # Display all contacts
    manager.display_contacts()
    
    # Find a contact
    contact = manager.find_contact("John Doe")
    if contact:
        print(f"\nFound contact: {contact.name}")

# Run the test
test_contact_manager()
```

### Exercise 2: Task Scheduler (Intermediate)
```python
from datetime import datetime, timedelta
from typing import List, Dict

class Task:
    def __init__(self, title: str, priority: int, deadline: datetime):
        self.title = title
        self.priority = priority  # 1 (highest) to 5 (lowest)
        self.deadline = deadline
        self.completed = False

class TaskScheduler:
    def __init__(self):
        self.tasks: List[Task] = []
    
    def add_task(self, title: str, priority: int, days_to_deadline: int) -> None:
        """Add a new task."""
        deadline = datetime.now() + timedelta(days=days_to_deadline)
        task = Task(title, priority, deadline)
        self.tasks.append(task)
        print(f"Added task: {title}")
    
    def get_urgent_tasks(self) -> List[Task]:
        """Get tasks due within 2 days and high priority (1-2)."""
        urgent = []
        two_days = datetime.now() + timedelta(days=2)
        
        for task in self.tasks:
            if not task.completed and task.deadline <= two_days and task.priority <= 2:
                urgent.append(task)
        
        return urgent
    
    def display_tasks(self, tasks: List[Task] = None) -> None:
        """Display tasks with their status."""
        if tasks is None:
            tasks = self.tasks
        
        if not tasks:
            print("No tasks found.")
            return
        
        print("\nTask List:")
        print("-" * 60)
        for task in sorted(tasks, key=lambda x: (x.priority, x.deadline)):
            status = "✓" if task.completed else " "
            days_left = (task.deadline - datetime.now()).days
            print(f"[{status}] {task.title}")
            print(f"    Priority: {task.priority}, Due in: {days_left} days")
            print("-" * 60)

# Test the task scheduler
def test_task_scheduler():
    scheduler = TaskScheduler()
    
    # Add some tasks
    scheduler.add_task("Complete project proposal", 1, 1)  # Urgent
    scheduler.add_task("Review code changes", 2, 2)      # Urgent
    scheduler.add_task("Update documentation", 3, 5)     # Not urgent
    
    # Display all tasks
    print("\nAll Tasks:")
    scheduler.display_tasks()
    
    # Display urgent tasks
    print("\nUrgent Tasks:")
    scheduler.display_tasks(scheduler.get_urgent_tasks())

# Run the test
test_task_scheduler()
```

## 6. Summary

### Key Takeaways
- Functions are essential for code organization and reusability
- Proper parameter handling and return values are crucial
- Modular programming helps manage complex applications
- Best practices focus on clarity, simplicity, and maintainability

### What's Next
- Working with Python modules and packages
- Object-Oriented Programming concepts
- Advanced error handling and debugging

---

> **Navigation**
> - [← Python Flow Control](02-Python-Flow-Control-Loops-Conditions.md)
> - [Python Modules and Packages →](04-Python-Modules-Packages.md)
