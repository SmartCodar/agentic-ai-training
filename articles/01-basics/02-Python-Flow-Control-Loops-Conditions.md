# Day 2 - Flow Control: If-Else, For Loop, While Loop

## Overview
Master Python's flow control structures to create dynamic and intelligent programs. Learn how to make decisions, iterate over data, and control program execution flow using Python's elegant syntax and powerful control structures.

## Theoretical Foundation

### Understanding Flow Control
Flow control is the cornerstone of programming logic, determining how a program executes instructions. In Python, flow control follows these key principles:

1. **Sequential Execution**: By default, code runs top to bottom
2. **Conditional Execution**: Code runs based on conditions (if-else)
3. **Iterative Execution**: Code repeats based on conditions (loops)
4. **Jump Statements**: Code execution can be altered (break, continue)

### Decision Making in Programming
Programs need to make decisions based on conditions, just like humans do:

1. **Boolean Logic**: True/False evaluations
2. **Comparison**: Testing relationships between values
3. **Branching**: Taking different paths based on conditions
4. **Pattern Matching**: Selecting actions based on data patterns (Python 3.10+)

### Iteration and Loops
Loops automate repetitive tasks and process collections of data:

1. **Counted Loops**: Known number of iterations (for loops)
2. **Conditional Loops**: Unknown iterations (while loops)
3. **Infinite Loops**: Continuous execution until interrupted
4. **Nested Loops**: Loops within loops for complex patterns

## ⏱️ Time Estimate
- **Reading**: 30 minutes
- **Exercises**: 45 minutes
- **Practice Project**: 30 minutes

## 🎯 Learning Objectives
By the end of this lesson, you will be able to:
- Implement conditional logic using if-elif-else statements
- Write efficient loops using for and while constructs
- Use pattern matching for elegant control flow (Python 3.11+)
- Apply loop control statements (break, continue, else)
- Debug common flow control issues

## 📋 Prerequisites
- Python 3.11+ installed ([Download Python](https://www.python.org/downloads/))
- Understanding of Python variables and operators (Day 1)
- VS Code with Python extension installed
- Basic understanding of Boolean logic

## 🛠️ Setup Check
Run this code to verify your Python installation and features:
```python
import sys

print(f"Python Version: {sys.version}")
print(f"Pattern Matching Available: {sys.version_info >= (3, 10)}")
```

---

## 1. Conditional Statements

### 1.1 Basic If-Else Statements
```python
# Simple if-else
def check_age(age: int) -> str:
    if age >= 18:
        return "Adult"
    else:
        return "Minor"

# Multiple conditions using elif
def grade_score(score: int) -> str:
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    else:
        return "F"

# Nested if statements
def validate_user(age: int, is_registered: bool) -> str:
    if age >= 18:
        if is_registered:
            return "Access granted"
        else:
            return "Please register"
    else:
        return "Age restricted"
```

### 1.2 Switch-like Flow Control

#### Traditional Dictionary Approach
```python
def get_day_type(day: str) -> str:
    return {
        'Monday': 'Workday',
        'Tuesday': 'Workday',
        'Wednesday': 'Workday',
        'Thursday': 'Workday',
        'Friday': 'Workday',
        'Saturday': 'Weekend',
        'Sunday': 'Weekend'
    }.get(day, 'Invalid day')

# With function callbacks
def process_command(command: str, data: dict) -> str:
    def create_item(): return f"Created {data.get('name', 'item')}"
    def update_item(): return f"Updated {data.get('id', 'unknown')}"
    def delete_item(): return f"Deleted {data.get('id', 'unknown')}"
    
    commands = {
        'create': create_item,
        'update': update_item,
        'delete': delete_item
    }
    
    action = commands.get(command.lower())
    return action() if action else 'Invalid command'
```

#### Modern Pattern Matching (Python 3.10+)
```python
def analyze_data(data: object) -> str:
    match data:
        case int() | float() as num if num > 0:
            return f"Positive number: {num}"
        case str() as text if text.strip():
            return f"Non-empty string: {text}"
        case list() | tuple() as seq if seq:
            return f"Non-empty sequence of length {len(seq)}"
        case dict() as d if d:
            return f"Dictionary with keys: {', '.join(d.keys())}"
        case _:
            return "Unhandled data type or empty value"

# Enhanced switch with pattern matching
def process_event(event: dict) -> str:
    match event:
        case {'type': 'user', 'action': 'login', 'id': id}:
            return f"User {id} logged in"
        case {'type': 'user', 'action': 'logout', 'id': id}:
            return f"User {id} logged out"
        case {'type': 'error', 'code': code, 'message': msg}:
            return f"Error {code}: {msg}"
        case {'type': 'system', **details}:
            return f"System event with details: {details}"
        case _:
            return "Unknown event"

# Example usage
print(analyze_data(42))          # "Positive number: 42"
print(analyze_data("Hello"))     # "Non-empty string: Hello"

event = {'type': 'user', 'action': 'login', 'id': 'user123'}
print(process_event(event))     # "User user123 logged in"
```

### 1.3 Advanced Conditionals
```python
# Walrus operator (Python 3.8+)
def process_data(data: list) -> tuple:
    if (n := len(data)) > 10:
        return data[:10], n
    return data, n

# Multiple conditions with any/all
def validate_password(password: str) -> bool:
    checks = [
        len(password) >= 8,
        any(c.isupper() for c in password),
        any(c.islower() for c in password),
        any(c.isdigit() for c in password)
    ]
    return all(checks)

# Ternary operator
def get_status(is_active: bool) -> str:
    return "Active" if is_active else "Inactive"
```

### 1.4 Best Practices
```python
# ✅ Use positive conditions when possible
if is_valid:  # Better than: if is_valid == True
    process()

# ✅ Combine related conditions
if 0 <= age <= 120:  # Better than: if age >= 0 and age <= 120
    validate_age()

# ✅ Use early returns for guard clauses
def process_user(user: dict) -> str:
    if not user:
        return "Invalid user"
    if not user.get('name'):
        return "Name required"
    
    # Main processing here
    return f"Processing user {user['name']}"
```

#### 1. User Authentication
```python
# Login system with password validation
def validate_login(username, password):
    stored_password = "secure123"  # In real apps, this would be hashed
    
    if not username or not password:  # Check for empty inputs
        return "Username and password are required"
    elif len(password) < 8:
        return "Password must be at least 8 characters"
    elif password == stored_password:
        return f"Welcome back, {username}!"
    else:
        return "Invalid credentials"

# Test the function
print(validate_login("john_doe", "secure123"))
```

#### 2. E-commerce Discount Calculator
```python
def calculate_discount(cart_total, is_member):
    if cart_total >= 1000 and is_member:
        discount = 0.20  # 20% discount
    elif cart_total >= 1000:
        discount = 0.10  # 10% discount
    elif cart_total >= 500 and is_member:
        discount = 0.05  # 5% discount
    else:
        discount = 0
    
    final_price = cart_total * (1 - discount)
    savings = cart_total - final_price
    
    return f"Final Price: ${final_price:.2f} (You save: ${savings:.2f})"

# Test different scenarios
print(calculate_discount(1200, True))   # Member with large purchase
print(calculate_discount(600, False))   # Non-member with medium purchase
```

#### 3. Weather Clothing Advisor
```python
def suggest_clothing(temperature, is_raining):
    if is_raining:
        if temperature < 50:
            return "Bring a warm raincoat and umbrella"
        else:
            return "Bring a light raincoat or umbrella"
    else:
        if temperature < 32:
            return "Wear a heavy winter coat"
        elif temperature < 50:
            return "Wear a light jacket"
        elif temperature < 70:
            return "Bring a sweater"
        else:
            return "T-shirt weather!"

# Test different weather conditions
print(suggest_clothing(45, True))   # Cold and rainy
print(suggest_clothing(75, False))  # Warm and sunny
```

---

## 3. For Loops

### Explanation
- Used for **iterating over a sequence** (list, tuple, dictionary, string).
- Executes a block of code for **each item**.

### Syntax
```python
for item in sequence:
    # code block
```

### Real-world Applications

#### 1. Data Processing
```python
# Calculate average rating for a product
product_ratings = [4.5, 5.0, 3.5, 4.0, 4.8, 4.2]

def analyze_ratings(ratings):
    total = 0
    count = 0
    
    for rating in ratings:
        total += rating
        count += 1
    
    average = total / count
    return f"Average Rating: {average:.1f} out of 5.0 ({count} reviews)"

print(analyze_ratings(product_ratings))
```

#### 2. Shopping Cart Total
```python
# Calculate cart total with tax
class CartItem:
    def __init__(self, name, price, quantity):
        self.name = name
        self.price = price
        self.quantity = quantity

def calculate_cart_total(items, tax_rate=0.08):
    subtotal = 0
    print("Shopping Cart Summary:")
    
    for item in items:
        item_total = item.price * item.quantity
        subtotal += item_total
        print(f"{item.name}: ${item_total:.2f} (${item.price:.2f} x {item.quantity})") 
    
    tax = subtotal * tax_rate
    total = subtotal + tax
    
    print(f"\nSubtotal: ${subtotal:.2f}")
    print(f"Tax ({tax_rate*100}%): ${tax:.2f}")
    print(f"Total: ${total:.2f}")

# Test with sample cart
cart = [
    CartItem("Laptop", 999.99, 1),
    CartItem("Mouse", 29.99, 2),
    CartItem("Keyboard", 59.99, 1)
]

calculate_cart_total(cart)
```

#### 3. File Word Counter
```python
def count_words(text):
    # Initialize counters
    total_words = 0
    long_words = 0  # words with 5+ characters
    
    # Process each line
    for line in text.split('\n'):
        # Skip empty lines
        if not line.strip():
            continue
            
        # Count words in the line
        words = line.split()
        total_words += len(words)
        
        # Count long words
        for word in words:
            if len(word) >= 5:
                long_words += 1
    
    return {
        'total_words': total_words,
        'long_words': long_words,
        'long_word_percentage': (long_words / total_words * 100) if total_words > 0 else 0
    }

# Test the function
sample_text = """Python is a versatile programming language.
It's great for beginners and experts alike.
Many developers use Python daily."""

stats = count_words(sample_text)
print(f"Total Words: {stats['total_words']}")
print(f"Long Words: {stats['long_words']}")
print(f"Long Word Percentage: {stats['long_word_percentage']:.1f}%")
```

---

## 2. Loops and Iteration

### 2.1 For Loops
```python
# Basic for loop with range
def generate_squares(n: int) -> list[int]:
    return [i ** 2 for i in range(n)]

# Iterating over sequences
def process_items(items: list) -> None:
    for index, item in enumerate(items):
        print(f"Processing item {index + 1}: {item}")

# Dictionary iteration
def analyze_data(data: dict) -> dict:
    summary = {
        'total': 0,
        'count': 0,
        'categories': set()
    }
    
    for category, values in data.items():
        summary['categories'].add(category)
        for value in values:
            summary['total'] += value
            summary['count'] += 1
    
    return summary

# Comprehensions (modern Pythonic way)
def process_numbers(numbers: list[int]) -> tuple[list, list, dict]:
    # List comprehension
    evens = [n for n in numbers if n % 2 == 0]
    
    # Set comprehension
    unique_squares = {n ** 2 for n in numbers}
    
    # Dict comprehension
    number_map = {n: bin(n) for n in numbers}
    
    return evens, list(unique_squares), number_map
```

### 2.2 While Loops
```python
# Basic while loop with counter
def countdown(start: int) -> None:
    while start > 0:
        print(f"T-minus {start}...")
        start -= 1
    print("Liftoff!")

# While loop with dynamic condition
def find_number(target: int, max_attempts: int = 5) -> tuple[bool, int]:
    import random
    attempts = 0
    
    while attempts < max_attempts:
        guess = random.randint(1, 100)
        attempts += 1
        
        if guess == target:
            return True, attempts
    
    return False, attempts

# Input validation with while
def get_valid_age() -> int:
    while True:
        try:
            age = int(input("Enter your age: "))
            if 0 <= age <= 120:
                return age
            print("Age must be between 0 and 120")
        except ValueError:
            print("Please enter a valid number")
```

### 2.3 Loop Control and Best Practices

#### Control Flow Statements
```python
# Using break and continue
def process_numbers(numbers: list[int]) -> tuple[list[int], list[int]]:
    evens = []
    odds = []
    
    for num in numbers:
        if num == 0:
            break  # Stop processing if we hit zero
        
        if num % 2 == 0:
            evens.append(num)
            continue  # Skip to next number
        
        odds.append(num)
    
    return evens, odds

# Loop with else clause
def find_element(items: list, target: any) -> int:
    for index, item in enumerate(items):
        if item == target:
            break
    else:  # Executed if no break occurs
        return -1
    return index
```

#### Best Practices
```python
# ✅ Use enumerate for counter
for i, item in enumerate(items, 1):  # Start count from 1
    print(f"Item {i}: {item}")

# ✅ Use zip for parallel iteration
for name, score in zip(names, scores):
    print(f"{name}: {score}")

# ✅ Use list comprehension for simple transformations
squares = [x**2 for x in range(10)]

# ✅ Use generator expressions for large sequences
sum(x**2 for x in range(1_000_000))  # Memory efficient

# ✅ Use dict/set comprehension when appropriate
name_lengths = {name: len(name) for name in names}
unique_letters = {char.lower() for char in text}
```

#### Common Pitfalls
```python
# ❌ Don't modify list while iterating
# Bad:
for item in items:
    if condition(item):
        items.remove(item)  # Can skip items!

# ✅ Good:
items = [item for item in items if not condition(item)]

# ❌ Don't use while True without break condition
# Bad:
while True:
    process_data()

# ✅ Good:
while True:
    if not process_data():
        break

# ❌ Don't use indices unless necessary
# Bad:
for i in range(len(items)):
    print(items[i])

# ✅ Good:
for item in items:
    print(item)
```

### 2.4 Real-World Examples

#### 1. Data Processing Pipeline
```python
def login_system():
    max_attempts = 3
    attempts = 0
    correct_password = "secure123"
    
    while attempts < max_attempts:
        password = input("Enter your password: ")
        if password == correct_password:
            print("Login successful!")
            return True
        else:
            attempts += 1
            remaining = max_attempts - attempts
            if remaining > 0:
                print(f"Incorrect password. {remaining} attempts remaining.")
            else:
                print("Account locked. Please contact support.")
    return False

# Test the function (commented out as it requires user input)
# login_system()
```

#### 2. Game Score Tracker
```python
def track_game_score():
    score = 0
    high_score = 100
    lives = 3
    
    while lives > 0:
        # Simulate playing a game level
        points = int(input("Enter points scored (-1 to quit): "))
        
        if points == -1:
            break
        
        score += points
        print(f"Current score: {score}")
        
        if score > high_score:
            print("New high score!")
            high_score = score
        
        # Simulate losing a life
        if points == 0:
            lives -= 1
            print(f"Lost a life! {lives} remaining")
    
    print(f"\nGame Over!")
    print(f"Final Score: {score}")
    print(f"High Score: {high_score}")

# Test the function (commented out as it requires user input)
# track_game_score()
```

#### 3. Data Validation Loop
```python
def get_valid_age():
    while True:
        try:
            age = int(input("Enter your age (1-120): "))
            
            if 1 <= age <= 120:
                return age
            else:
                print("Age must be between 1 and 120")
        except ValueError:
            print("Please enter a valid number")

def get_valid_email():
    import re
    
    while True:
        email = input("Enter your email: ")
        # Simple email validation pattern
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if re.match(pattern, email):
            return email
        else:
            print("Invalid email format. Please try again.")

# Example usage (commented out as it requires user input)
# age = get_valid_age()
# email = get_valid_email()
# print(f"Validated Age: {age}")
# print(f"Validated Email: {email}")
```

---

## 5. Best Practices and Code Standards

### Conditional Statements Best Practices

#### ✅ DO: Use positive conditions when possible
```python
# Good
if is_valid and is_active:
    process_user()

# Avoid
if not is_invalid and not is_inactive:
    process_user()
```

#### ✅ DO: Use early returns for guard clauses
```python
def process_payment(amount):
    # Guard clauses
    if amount <= 0:
        return "Invalid amount"
    if not is_account_active:
        return "Account inactive"
    
    # Main logic
    return process_transaction(amount)
```

#### ❌ DON'T: Use nested if statements when unnecessary
```python
# Bad - deeply nested
if condition1:
    if condition2:
        if condition3:
            do_something()

# Good - use logical operators
if condition1 and condition2 and condition3:
    do_something()
```

### Loop Best Practices

#### ✅ DO: Use enumerate() for counter and value
```python
# Good
for index, value in enumerate(items):
    print(f"Item {index + 1}: {value}")

# Avoid
index = 0
for value in items:
    print(f"Item {index + 1}: {value}")
    index += 1
```

#### ✅ DO: Use list comprehension for simple transformations
```python
# Good
squares = [x**2 for x in range(10)]

# Avoid
squares = []
for x in range(10):
    squares.append(x**2)
```

#### ❌ DON'T: Create infinite loops without exit conditions
```python
# Bad - might never end
while True:
    do_something()

# Good - clear exit condition
while attempts < max_attempts:
    do_something()
    attempts += 1
```

### Common Pitfalls to Avoid

#### ❌ DON'T: Forget to increment loop counters
```python
# Bad - infinite loop
count = 0
while count < 5:
    print(count)
    # Forgot to increment count

# Good
count = 0
while count < 5:
    print(count)
    count += 1
```

#### ❌ DON'T: Use break/continue without clear comments
```python
# Bad - unclear why we're breaking
for item in items:
    if item.value < 0:
        break

# Good - clear intention
for item in items:
    if item.value < 0:
        break  # Stop processing at first negative value
```

### Performance Tips

#### ✅ DO: Use appropriate loop type
```python
# Good - for loop when sequence length is known
for i in range(len(items)):
    process_item(items[i])

# Good - while loop when end condition is unknown
while not found_match:
    item = get_next_item()
    if is_match(item):
        found_match = True
```

#### ✅ DO: Consider using itertools for complex iterations
```python
from itertools import combinations

# Efficient way to get all pairs
pairs = list(combinations(items, 2))
```

---

## 6. Hands-on Practice

### Exercise 1: Temperature Converter (Basic)
```python
def convert_temperature():
    """Temperature converter with input validation and multiple conversions."""
    while True:
        print("\nTemperature Converter")
        print("1. Celsius to Fahrenheit")
        print("2. Fahrenheit to Celsius")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '3':
            print("Goodbye!")
            break
            
        try:
            temp = float(input("Enter temperature: "))
            
            if choice == '1':
                converted = (temp * 9/5) + 32
                print(f"{temp}°C = {converted:.1f}°F")
            elif choice == '2':
                converted = (temp - 32) * 5/9
                print(f"{temp}°F = {converted:.1f}°C")
            else:
                print("Invalid choice!")
        except ValueError:
            print("Please enter a valid number!")

# Test the program
# convert_temperature()
```

### Exercise 2: Task Manager (Intermediate)
```python
def task_manager():
    """Simple task manager with CRUD operations."""
    tasks = []
    
    while True:
        print("\nTask Manager")
        print("1. Add Task")
        print("2. View Tasks")
        print("3. Mark Task Complete")
        print("4. Delete Task")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            task = input("Enter task description: ")
            tasks.append({"description": task, "completed": False})
            print("Task added successfully!")
            
        elif choice == '2':
            if not tasks:
                print("No tasks found!")
            else:
                print("\nCurrent Tasks:")
                for i, task in enumerate(tasks, 1):
                    status = "✓" if task["completed"] else " "
                    print(f"{i}. [{status}] {task['description']}")
                    
        elif choice == '3':
            if not tasks:
                print("No tasks to mark complete!")
            else:
                try:
                    task_num = int(input("Enter task number to mark complete: ")) - 1
                    if 0 <= task_num < len(tasks):
                        tasks[task_num]["completed"] = True
                        print("Task marked complete!")
                    else:
                        print("Invalid task number!")
                except ValueError:
                    print("Please enter a valid number!")
                    
        elif choice == '4':
            if not tasks:
                print("No tasks to delete!")
            else:
                try:
                    task_num = int(input("Enter task number to delete: ")) - 1
                    if 0 <= task_num < len(tasks):
                        deleted = tasks.pop(task_num)
                        print(f"Deleted task: {deleted['description']}")
                    else:
                        print("Invalid task number!")
                except ValueError:
                    print("Please enter a valid number!")
                    
        elif choice == '5':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice!")

# Test the program
# task_manager()
```

### Exercise 3: Number Guessing Game (Advanced)
```python
def number_guessing_game():
    """Enhanced number guessing game with difficulty levels and score tracking."""
    import random
    
    def get_difficulty():
        while True:
            print("\nSelect Difficulty:")
            print("1. Easy (1-50)")
            print("2. Medium (1-100)")
            print("3. Hard (1-200)")
            choice = input("Enter choice (1-3): ")
            
            if choice in ['1', '2', '3']:
                return int(choice)
            print("Invalid choice!")
    
    def calculate_score(attempts, max_attempts):
        return max(0, 100 * (max_attempts - attempts + 1) // max_attempts)
    
    while True:
        # Game setup
        difficulty = get_difficulty()
        max_number = {1: 50, 2: 100, 3: 200}[difficulty]
        max_attempts = {1: 10, 2: 7, 3: 5}[difficulty]
        target = random.randint(1, max_number)
        attempts = 0
        guesses = []
        
        print(f"\nI'm thinking of a number between 1 and {max_number}")
        print(f"You have {max_attempts} attempts to guess it!")
        
        # Main game loop
        while attempts < max_attempts:
            try:
                guess = int(input(f"\nAttempt {attempts + 1}/{max_attempts}. Your guess: "))
                
                if guess in guesses:
                    print("You already tried that number!")
                    continue
                    
                guesses.append(guess)
                attempts += 1
                
                if guess == target:
                    score = calculate_score(attempts, max_attempts)
                    print(f"\nCongratulations! You got it in {attempts} tries!")
                    print(f"Score: {score} points")
                    break
                elif guess < target:
                    print("Too low!")
                else:
                    print("Too high!")
                    
                if attempts < max_attempts:
                    print(f"Attempts remaining: {max_attempts - attempts}")
                    
            except ValueError:
                print("Please enter a valid number!")
        
        if attempts >= max_attempts and guess != target:
            print(f"\nGame Over! The number was {target}")
        
        if input("\nPlay again? (y/n): ").lower() != 'y':
            print("Thanks for playing!")
            break

# Test the program
# number_guessing_game()
```

---

## 7. Summary

### Key Takeaways
- Conditional statements (`if-else`) enable decision-making in programs
- `for` loops are ideal for iterating over sequences with known lengths
- `while` loops handle repetition based on conditions
- Best practices focus on readability, maintainability, and efficiency

### What's Next
- Functions and modular programming
- Code organization and reusability
- Parameter passing and return values

---

## 📚 Additional Resources
- [Python Control Flow Documentation](https://docs.python.org/3/tutorial/controlflow.html)
- [PEP 634 – Structural Pattern Matching](https://peps.python.org/pep-0634/)
- [Python Tips: Loop Better](https://book.pythontips.com/en/latest/for_-_else.html)
- [Real Python: Python For Loops](https://realpython.com/python-for-loop/)

## ✅ Knowledge Check
1. What is the difference between `break` and `continue` statements?
2. When should you use a `while` loop instead of a `for` loop?
3. How does pattern matching improve code readability compared to if-elif chains?
4. What are the benefits of using list comprehensions over traditional for loops?
5. How does the loop `else` clause work and when is it useful?

## 🔍 Common Issues and Solutions
| Issue | Solution |
|-------|----------|
| Infinite Loop | Always ensure loop condition can become False |
| List Modification | Use list comprehension or iterate over copy |
| Index Errors | Use enumerate instead of manual indexing |
| Memory Issues | Use generators for large sequences |

## 📝 Summary
- Flow control enables dynamic program behavior through conditions and loops
- Pattern matching (Python 3.10+) provides elegant handling of complex conditions
- Modern Python features like walrus operator and f-strings improve code clarity
- Best practices focus on readability and performance
- Loop control statements provide fine-grained iteration control

> **Navigation**
> - [← Variables and Data Types](01-Python-Basics-Variables-Types-Operators.md)
> - [Functions and Modularity →](03-Python-Functions-Modular-Programming.md)
