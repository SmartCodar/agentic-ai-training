# Day 1 - Python Basics: Variables, Data Types, Operators

## Overview
This lesson covers fundamental Python concepts that form the building blocks of any Python program. Understanding these basics is crucial for becoming proficient in Python programming.

## Learning Objectives
- Master Python variable creation and naming conventions
- Understand and work with different Python data types
- Learn to use basic Python operators
- Apply Python coding best practices

## Prerequisites
- Basic computer literacy
- Python 3.x installed on your computer
- A text editor or IDE (VS Code recommended)
- Basic understanding of what programming is

## Time Estimate
- Reading: 20 minutes
- Practice: 40 minutes
- Assignments: 30 minutes

---

## 1. Introduction
Welcome to Day 1! Today, we will:
- Understand Variables and Naming Rules
- Explore Core Data Types in Python
- Perform Basic Operations using Operators

By the end, you will be comfortable creating variables, handling data types, and using basic operators.

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

## 3. Data Types

### Core Data Types
| Type    | Example               | Description                       |
|---------|-----------------------|-----------------------------------|
| `int`   | `5`, `-10`, `0`         | Whole numbers                    |
| `float` | `3.14`, `-0.99`, `0.0`  | Decimal numbers                  |
| `str`   | `'hello'`, `"world"`    | Text                             |
| `bool`  | `True`, `False`         | Boolean values                   |
| `list`  | `[1, 2, 3]`             | Ordered, changeable collection   |
| `tuple` | `(1, 2, 3)`             | Ordered, unchangeable collection |
| `dict`  | `{"name": "John"}`     | Key-value pairs                  |
| `set`   | `{1, 2, 3}`             | Unordered, unique elements       |

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

## 4. Operators

### Arithmetic Operators
```python
a = 10
b = 3

print(a + b)  # 13
print(a - b)  # 7
print(a * b)  # 30
print(a / b)  # 3.3333
print(a // b) # 3
print(a % b)  # 1
print(a ** b) # 1000
```

### Comparison Operators
```python
x = 5
y = 7

print(x == y)   # False
print(x != y)   # True
print(x > y)    # False
print(x <= y)   # True
```

### Logical Operators
```python
p = True
q = False

print(p and q)  # False
print(p or q)   # True
print(not p)    # False
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

> **Next Up:** Day 2 - Flow Control (If-Else, For Loop, While Loop)
