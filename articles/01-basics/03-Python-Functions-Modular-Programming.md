# Day 3 - Functions and Modular Programming: Building Reusable Code

## Overview
This lesson focuses on functions and modular programming in Python, teaching you how to write reusable, organized, and maintainable code. Functions are the building blocks of modular programming, allowing you to break down complex problems into smaller, manageable pieces.

## Learning Objectives
- Master function creation and usage in Python
- Understand parameter passing and return values
- Learn about function scope and variable visibility
- Apply modular programming principles
- Implement error handling in functions

## Prerequisites
- Completion of [Python Basics](01-Python-Basics-Variables-Types-Operators.md)
- Completion of [Flow Control](02-Python-Flow-Control-Loops-Conditions.md)
- Python 3.x installed on your computer


## 1. Introduction to Functions

### What is a Function?
A function is a reusable block of code that performs a specific task. Functions help organize code, promote reusability, and make programs easier to understand and maintain.

### Basic Function Syntax
```python
def greet():
    print("Hello, World!")

# Call the function
greet()
```

### Parameters and Return Values
```python
def add_numbers(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b

# Call function with parameters
result = add_numbers(5, 3)
print(f"Sum: {result}")
```

### Default Parameters
```python
def greet_user(name: str = "Guest") -> str:
    """Greet a user with optional name parameter."""
    return f"Hello, {name}!"

# Using default parameter
print(greet_user())          # Output: Hello, Guest!
print(greet_user("Alice"))   # Output: Hello, Alice!
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
