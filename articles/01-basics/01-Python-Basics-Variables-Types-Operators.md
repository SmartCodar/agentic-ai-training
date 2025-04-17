# Day 1 - Python Basics: Variables, Data Types, Operators

## Overview
Master the foundational elements of Python programming. This comprehensive lesson covers Python's core building blocks: variables, data types, and operators. You'll learn how to write clean, efficient code following industry best practices and PEP 8 standards.

## 🎯 Learning Objectives
By the end of this lesson, you will be able to:
- Create and manage variables following Python naming conventions
- Work with Python's core data types (int, float, str, bool, list, dict)
- Apply arithmetic, comparison, and logical operators effectively
- Implement Python coding best practices and PEP 8 standards
- Debug common variable and data type issues

## 📋 Prerequisites
- Python 3.11+ installed ([Download Python](https://www.python.org/downloads/))
- Code editor (VS Code recommended) with Python extension
- Basic understanding of programming concepts

## Theoretical Foundation

### What is Python?
Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. Its design philosophy emphasizes code readability with the use of significant whitespace and simple, clear syntax. Python supports multiple programming paradigms, including:

- **Procedural Programming**: Writing code as a sequence of steps
- **Object-Oriented Programming**: Organizing code into objects that contain data and code
- **Functional Programming**: Treating computation as the evaluation of mathematical functions

### Why Python?
Python has become one of the world's most popular programming languages because:

1. **Readability**: Python's syntax is clear and expressive, making it easy to understand and maintain
2. **Versatility**: It's used in web development, data science, AI, automation, and more
3. **Rich Ecosystem**: Vast collection of libraries and frameworks for various applications
4. **Community Support**: Large, active community providing resources and help
5. **Industry Adoption**: Widely used by companies like Google, Netflix, and NASA

### How Python Works
Understanding how Python works under the hood helps in writing better code:

1. **Source Code**: You write Python code in `.py` files
2. **Compilation**: Python compiles your code to bytecode (`.pyc` files)
3. **Interpretation**: Python Virtual Machine (PVM) executes the bytecode
4. **Memory Management**: Python automatically handles memory allocation and garbage collection
5. **Dynamic Typing**: Variables are checked at runtime, not compile time

### Python's Building Blocks
Before diving into the practical aspects, let's understand the three fundamental building blocks we'll cover today:

1. **Variables**: Containers that store data values
   - Act as references to memory locations
   - Can hold different types of data
   - Names are case-sensitive and must follow rules

2. **Data Types**: Define the kind of data a variable can hold
   - Determine what operations can be performed
   - Affect memory usage and performance
   - Can be mutable or immutable

3. **Operators**: Symbols that perform operations on variables and values
   - Transform and combine data
   - Control program flow
   - Follow specific precedence rules

---

## 1. Python Variables and Data Management

### 1.1 Understanding Variables
Variables in Python are dynamic references to memory locations that store data. Unlike statically-typed languages, Python uses dynamic typing and type inference.

```python
# Dynamic typing in action
x = 42          # x is an integer
print(type(x))  # <class 'int'>

x = "Hello"     # x is now a string
print(type(x))  # <class 'str'>
```

### 1.2 Variable Naming Conventions
Python uses specific naming conventions defined in PEP 8:

```python
# Correct variable naming
user_name = "John"          # Snake case for variables
MAX_ATTEMPTS = 3           # Upper case for constants
TotalStudents = 100        # Pascal case for classes

# Invalid names
2nd_place = "Silver"       # Can't start with number
user-name = "John"        # Can't use hyphens
class = "Python"          # Can't use reserved words
```

### 1.3 Memory Management
Understand how Python manages variable memory:

```python
# Memory reference example
a = [1, 2, 3]
b = a           # b references same list as a

b.append(4)     # Modifies the shared list
print(a)        # [1, 2, 3, 4]
print(b)        # [1, 2, 3, 4]

# Creating a copy
c = a.copy()    # c is a new list
c.append(5)     # Only modifies c
print(a)        # [1, 2, 3, 4]
print(c)        # [1, 2, 3, 4, 5]
```

---

## 2. Variables

### Explanation
- Variables store information for later use.
- Python uses **dynamic typing**: you don't declare types explicitly.
- **Naming Rules:**
  - Must start with a letter (a-z, A-Z) or underscore (_)
  - Cannot start with a number
  - Case-sensitive (`name`, `Name`, and `NAME` are different)

### Examples
```python
# Basic variable assignment
name = "Alice"
age = 22
height = 5.4
is_student = True

print(name, age, height, is_student)

# Updating variables
score = 90
score = 95
print(score)

# Dynamic typing example
x = 5       # int
y = "five"  # now a string
print(type(x), type(y))
```

### Common Pitfalls and Solutions
- ❌ Using cryptic names like `tp`
- ✅ Using descriptive names like `total_price`
- ❌ Using camelCase: `finalScore`
- ✅ Using snake_case: `final_score`

### Real-world Application
```python
# E-commerce example
product_name = "Laptop"
product_price = 999.99
quantity_in_stock = 50
is_available = True

# Calculate total value of inventory
inventory_value = product_price * quantity_in_stock
print(f"Total inventory value: ${inventory_value}")
```

---

## 3. Python Data Types

### 3.1 Core Data Types Overview
| Type    | Example               | Mutability | Use Case                         |
|---------|-----------------------|------------|----------------------------------|
| `int`   | `5`, `-10`, `0`       | Immutable  | Counting, indexing, math         |
| `float` | `3.14`, `-0.99`       | Immutable  | Scientific calculations, prices   |
| `str`   | `'hello'`, `"world"`  | Immutable  | Text processing, data formatting |
| `bool`  | `True`, `False`       | Immutable  | Control flow, flags              |
| `list`  | `[1, 2, 3]`           | Mutable    | Sequential data, collections     |
| `tuple` | `(1, 2, 3)`           | Immutable  | Fixed collections, coordinates   |
| `dict`  | `{"name": "John"}`   | Mutable    | Key-value mapping, JSON data     |
| `set`   | `{1, 2, 3}`           | Mutable    | Unique items, fast lookups       |

### 3.2 Working with Numbers
```python
# Integer operations
age = 25
count = 1_000_000  # Use underscores for readability
hex_value = 0xFF   # Hexadecimal
bin_value = 0b1010 # Binary

# Float operations and precision
price = 19.99
pi = 3.14159
scientific = 2.5e-3  # Scientific notation

# Avoiding floating-point precision issues
from decimal import Decimal

total = Decimal('19.99') * Decimal('0.15')  # Precise decimal calculations
```

### 3.3 String Operations
```python
# String creation and formatting
name = 'Alice'
greeting = f"Hello, {name}!"  # f-strings (Python 3.6+)
message = "Hello, {}!".format(name)  # .format() method

# String methods
text = "  Python Programming  "
print(text.strip())         # Remove whitespace
print(text.lower())        # Convert to lowercase
print(text.split())        # Split into list

# Multi-line strings
doc = """This is a
    multi-line string that
    preserves formatting."""
```

### 3.4 Collections
```python
# List operations
fruits = ['apple', 'banana', 'orange']
fruits.append('grape')     # Add item
fruits.insert(0, 'kiwi')  # Insert at position
fruits.sort()             # Sort in place

# Tuple operations
point = (3, 4)
x, y = point              # Tuple unpacking
coordinates = (*point, 5) # Tuple expansion

# Dictionary operations
user = {
    'name': 'John Doe',
    'age': 30,
    'is_active': True
}

# Dict comprehension
squares = {x: x**2 for x in range(5)}

# Set operations
valid_users = {'alice', 'bob', 'charlie'}
active_users = {'alice', 'charlie', 'david'}

# Set operations
common_users = valid_users & active_users  # Intersection
all_users = valid_users | active_users     # Union
```

### 3.5 Type Conversion
```python
# Explicit type conversion
num_str = "123"
num_int = int(num_str)    # String to integer
num_float = float(num_str) # String to float

# Collection conversions
num_list = [1, 2, 2, 3, 3, 4]
num_set = set(num_list)   # Convert to set (removes duplicates)
num_tuple = tuple(num_set) # Convert to tuple

# Working with binary data
bytes_data = bytes([65, 66, 67])  # Creates b'ABC'
bytearray_data = bytearray(bytes_data)  # Mutable bytes
```

### Examples and Use Cases
```python
# Basic types with real-world context
user_name = "John"              # str: User profile information
user_age = 25                  # int: Age calculation
bank_balance = 1250.75         # float: Financial calculations
is_premium_user = True         # bool: Feature access control

# Print types and values with f-strings
print(f"Name ({type(user_name)}): {user_name}")
print(f"Age ({type(user_age)}): {user_age}")

# Complex types with practical applications
# List: Shopping cart items
cart_items = ["Laptop", "Mouse", "Keyboard"]

# Tuple: Geographic coordinates (immutable)
location = (40.7128, -74.0060)

# Dict: User profile
user_profile = {
    "id": 101,
    "name": "Alice",
    "email": "alice@example.com"
}

# Set: Unique tags on a blog post
post_tags = {"python", "programming", "tutorial", "python"}

print(f"\nCart Items: {cart_items}")
print(f"Location: {location}")
print(f"User Profile: {user_profile}")
print(f"Unique Tags: {post_tags}")
```

---

## 4. Python Operators

### 4.1 Arithmetic Operators
```python
# Basic arithmetic
a, b = 10, 3

sum_result = a + b        # Addition: 13
diff_result = a - b       # Subtraction: 7
prod_result = a * b       # Multiplication: 30
div_result = a / b        # Division: 3.3333...
floor_div = a // b        # Floor division: 3
modulus = a % b           # Modulus: 1
power = a ** b            # Exponentiation: 1000

# Augmented assignment
total = 0
total += 5                # Add and assign
total *= 2                # Multiply and assign
total /= 4                # Divide and assign

# Real-world example: Calculate discount
price = 100
discount_percent = 20
discount_amount = price * (discount_percent / 100)
final_price = price - discount_amount
```

### 4.2 Comparison Operators
```python
# Basic comparisons
x, y = 5, 7

equal = x == y           # Equal to: False
not_equal = x != y       # Not equal to: True
greater = x > y          # Greater than: False
less = x < y             # Less than: True
greater_equal = x >= y   # Greater than or equal: False
less_equal = x <= y      # Less than or equal: True

# String comparisons
name1 = "Alice"
name2 = "Bob"
print(name1 < name2)      # True (alphabetical comparison)

# Real-world example: Age verification
age = 18
is_adult = age >= 18     # True
can_vote = age >= 18     # True
```

### 4.3 Logical Operators
```python
# Basic logical operations
is_valid = True
is_active = False

and_result = is_valid and is_active    # False
or_result = is_valid or is_active      # True
not_result = not is_valid              # False

# Short-circuit evaluation
default_value = None
user_input = ""
valid_input = user_input or default_value  # None

# Real-world example: User authentication
has_account = True
is_logged_in = True
is_admin = False

can_view_admin = has_account and is_logged_in and is_admin  # False
can_view_content = has_account and (is_logged_in or is_admin)  # True
```

### 4.4 Bitwise Operators
```python
# Bitwise operations
a = 0b1100  # 12 in binary
b = 0b1010  # 10 in binary

bitwise_and = a & b      # AND: 0b1000 (8)
bitwise_or = a | b       # OR:  0b1110 (14)
bitwise_xor = a ^ b      # XOR: 0b0110 (6)
bitwise_not = ~a         # NOT: -13
left_shift = a << 1      # Left shift: 0b11000 (24)
right_shift = a >> 1     # Right shift: 0b0110 (6)

# Real-world example: Flag operations
READ = 0b100    # 4
WRITE = 0b010   # 2
EXEC = 0b001    # 1

# Setting permissions
permissions = READ | WRITE  # 0b110 (6)

# Checking permissions
can_read = permissions & READ == READ    # True
can_write = permissions & WRITE == WRITE  # True
can_exec = permissions & EXEC == EXEC     # False
```

### 4.5 Identity and Membership Operators
```python
# Identity operators
list1 = [1, 2, 3]
list2 = [1, 2, 3]
list3 = list1

print(list1 is list3)      # True (same object)
print(list1 is list2)      # False (different objects)
print(list1 is not list2)  # True

# Membership operators
fruits = ['apple', 'banana', 'orange']
print('apple' in fruits)     # True
print('grape' not in fruits) # True

# Real-world example: Menu options
MENU_OPTIONS = {'view', 'edit', 'delete'}
user_action = 'edit'

if user_action in MENU_OPTIONS:
    print(f"Processing {user_action} operation...")
else:
    print("Invalid operation!")
```

---

## 5. Best Practices and Standards

### Code Style Guidelines
```python
# ✅ Good Practice
user_age = 25
total_price = 99.99
is_active = True

# ❌ Bad Practice
a = 25          # Non-descriptive name
TotalPrice = 99.99  # Not snake_case
isactive = True    # Inconsistent naming
```

### DOs:
✅ Use descriptive variable names
```python
monthly_revenue = 5000  # Clear purpose
total_users = 100       # Self-explanatory
```

✅ Follow consistent naming conventions
```python
first_name = "John"     # snake_case for variables
MAX_ATTEMPTS = 3       # UPPERCASE for constants
```

✅ Add meaningful comments
```python
# Calculate discount for premium users
discount = total_price * 0.2 if is_premium else 0
```

### DON'Ts:
❌ Don't use reserved keywords
```python
# Bad - 'list' is a built-in type
list = [1, 2, 3]  # Don't do this!

# Good - use a descriptive name
number_list = [1, 2, 3]
```

❌ Don't use cryptic abbreviations
```python
# Bad
fn = "John"     # What is fn?

# Good
first_name = "John"  # Clear and readable
```

❌ Don't leave unused variables
```python
# Bad - unused variable
temp = calculate_total()
final_total = calculate_total() * 1.1

# Good - use what you declare
final_total = calculate_total() * 1.1
```

---

## 6. Hands-on Practice

### Exercise 1: Student Profile (Basic)
```python
# Create a student profile with different data types
student_name = "Your Name"  # Replace with your name
college_name = "Your College"  # Replace with your college
final_grade = 85.5
is_passed = True

# Print profile with proper formatting
print(f"Student Profile:")
print(f"Name: {student_name} (Type: {type(student_name)})") 
print(f"College: {college_name} (Type: {type(college_name)})")
print(f"Grade: {final_grade} (Type: {type(final_grade)})")
print(f"Passed: {is_passed} (Type: {type(is_passed)})")
```

### Exercise 2: Smart Calculator (Intermediate)
```python
# Create an interactive calculator
def calculator():
    # Get user input
    num1 = float(input("Enter first number: "))
    num2 = float(input("Enter second number: "))
    
    # Perform calculations
    operations = {
        "Addition": num1 + num2,
        "Subtraction": num1 - num2,
        "Multiplication": num1 * num2,
        "Division": num1 / num2 if num2 != 0 else "Error: Division by zero",
        "Floor Division": num1 // num2 if num2 != 0 else "Error: Division by zero",
        "Modulo": num1 % num2 if num2 != 0 else "Error: Division by zero",
        "Exponentiation": num1 ** num2
    }
    
    # Display results
    print("\nResults:")
    for operation, result in operations.items():
        print(f"{operation}: {result}")

# Run the calculator
calculator()
```

### Exercise 3: Grade Analyzer (Advanced)
```python
# Create a grade analyzer
def analyze_grades():
    # Get scores
    math_score = float(input("Enter math score (0-100): "))
    science_score = float(input("Enter science score (0-100): "))
    
    # Calculate average
    average = (math_score + science_score) / 2
    
    # Determine performance level
    if average > 90:
        performance = "Outstanding"
    elif average > 75:
        performance = "Excellent"
    elif average > 60:
        performance = "Good"
    else:
        performance = "Needs Improvement"
    
    # Display results
    print(f"\nResults:")
    print(f"Math Score: {math_score}")
    print(f"Science Score: {science_score}")
    print(f"Average: {average:.2f}")
    print(f"Performance: {performance}")

# Run the grade analyzer
analyze_grades()
```

### Bonus Challenge: E-commerce Cart (Expert)
```python
# Create a simple e-commerce cart system
def process_cart():
    # Product catalog (dictionary)
    products = {
        "laptop": 999.99,
        "mouse": 29.99,
        "keyboard": 59.99
    }
    
    # Shopping cart (list of dictionaries)
    cart = []
    
    # Add items to cart
    while True:
        print("\nAvailable products:")
        for product, price in products.items():
            print(f"{product}: ${price}")
        
        item = input("\nEnter product name (or 'done' to checkout): ").lower()
        if item == 'done':
            break
        
        if item in products:
            quantity = int(input(f"Enter quantity for {item}: "))
            cart.append({
                "product": item,
                "price": products[item],
                "quantity": quantity
            })
        else:
            print("Product not found!")
    
    # Calculate total
    total = sum(item["price"] * item["quantity"] for item in cart)
    
    # Display receipt
    print("\n====== Receipt ======")
    for item in cart:
        subtotal = item["price"] * item["quantity"]
        print(f"{item['product']} x{item['quantity']}: ${subtotal:.2f}")
    print(f"\nTotal: ${total:.2f}")

# Run the cart system
process_cart()
```

---

## 7. Summary

Today you learned:
- How to create and use variables
- Different types of data in Python
- Performing operations using basic operators
- Following coding best practices

You are now ready to move to control structures in Python (if-else, loops)!

---

## 🔍 Common Issues and Solutions
| Issue | Solution |
|-------|----------|
| `NameError: name 'x' is not defined` | Ensure variable is defined before use |
| `TypeError: can only concatenate str (not "int") to str` | Convert numbers to strings when concatenating |
| `SyntaxError: invalid syntax` | Check for missing quotes or parentheses |
| `IndentationError` | Use consistent indentation (4 spaces recommended) |

## 📚 Additional Resources
- [Python Official Documentation - Built-in Types](https://docs.python.org/3/library/stdtypes.html)
- [PEP 8 Style Guide](https://peps.python.org/pep-0008/)
- [Python Data Types - Real Python](https://realpython.com/python-data-types/)
- [Python Variables - W3Schools](https://www.w3schools.com/python/python_variables.asp)

## ✅ Knowledge Check
1. What is the difference between mutable and immutable data types in Python?
2. Why should we use snake_case for variable names in Python?
3. How does Python's dynamic typing differ from static typing?
4. When should you use a tuple instead of a list?
5. What are the benefits of using f-strings for string formatting?

> **Navigation**
> - [← Course Overview](../README.md)
> - [Flow Control →](02-Python-Flow-Control-Loops-Conditions.md)
