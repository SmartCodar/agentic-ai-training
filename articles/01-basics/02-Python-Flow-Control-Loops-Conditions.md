# Day 2 - Flow Control: If-Else, For Loop, While Loop

## Overview
This lesson covers Python's flow control structures that allow you to make decisions and repeat actions in your programs. These concepts are fundamental to creating dynamic and responsive applications.

## Learning Objectives
- Master conditional statements using if-else
- Understand and implement different types of loops
- Learn when to use for loops vs while loops
- Apply flow control concepts in real-world scenarios

## Prerequisites
- Understanding of Python variables and data types
- Basic operators and expressions in Python
- Python 3.x installed on your computer
- Basic command line knowledge

## Time Estimate
- Reading: 25 minutes
- Practice: 45 minutes
- Assignments: 35 minutes

---

## 1. Introduction
Today we focus on **Decision Making** and **Loops** in Python:
- How programs make choices (if-else)
- How programs repeat actions (for-loop, while-loop)

Flow control is the **backbone of programming logic**!

---

## 2. If-Else Statements

### Explanation
- Allows decisions based on conditions.
- Python uses indentation to define the block.

### Syntax
```python
if condition:
    # code block
elif another_condition:
    # code block
else:
    # code block
```

### Real-world Examples

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

## 4. While Loops

### Explanation
- Repeats code **as long as condition is True**.
- Caution: Make sure the loop eventually stops (update condition inside loop).

### Syntax
```python
while condition:
    # code block
```

### Real-world Applications

#### 1. Password Retry System
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

> **Navigation**
> - [← Day 1 - Python Basics](Day-1.md)
> - [Day 3 - Functions and Modular Programming →](Day-3.md)
