# Day 4 - Python Modules and Packages

## Overview

Discover how to organize, reuse, and scale your Python projects effectively using **modules and packages**. This lesson covers the structure, creation, import patterns, and usage of Python modules and packages to build clean, modular codebases.

## 🌟 Learning Objectives

By the end of this lesson, you will be able to:

- Understand the difference between modules and packages
- Create and import your own Python modules
- Organize files using packages and sub-packages
- Use built-in and third-party modules
- Apply relative and absolute imports
- Avoid circular import issues

## 📋 Prerequisites

- Completion of Days 1-3 (variables, data types, control flow)
- Python 3.11+ installed ([Download Python](https://www.python.org/downloads/))
- VS Code or any modern IDE

---

## 1. Theoretical Foundation

### 1.1 What is a Module?

A **module** is a single `.py` file that contains Python code like functions, classes, or variables. Modules help you split complex code into manageable chunks.

```python
# greetings.py

def say_hello(name):
    return f"Hello, {name}!"
```

### 1.2 What is a Package?

A **package** is a directory containing a special `__init__.py` file and one or more module files. Packages allow logical grouping of related modules.

```
my_package/
├── __init__.py
├── utils.py
└── config.py
```

---

## 2. Creating and Importing Modules

### 2.1 Creating Your Own Module

```python
# math_utils.py
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
```

### 2.2 Using Custom Module

```python
# main.py
import math_utils

print(math_utils.add(5, 3))         # Output: 8
print(math_utils.multiply(2, 4))    # Output: 8
```

### 2.3 Selective Import

```python
from math_utils import add
print(add(10, 20))
```

### 2.4 Renaming Modules

```python
import math_utils as mu
print(mu.add(1, 2))
```

---

## 3. Built-in and Third-Party Modules

### 3.1 Python Standard Library

```python
import os
print(os.getcwd())

import datetime
print(datetime.datetime.now())
```

### 3.2 Installing Third-party Packages

```bash
pip install requests
```

```python
import requests
response = requests.get("https://httpbin.org/get")
print(response.status_code)
```

---

## 4. Package Structure and Imports

### 4.1 Basic Package

```
myapp/
├── __init__.py
├── services.py
└── database.py
```

```python
# services.py
from .database import connect
```

### 4.2 Absolute vs Relative Imports

```python
# Absolute
from myapp.database import connect

# Relative
from .database import connect
```

### 4.3 Init File Role

`__init__.py` makes a directory a Python package. You can use it to expose APIs.

```python
# __init__.py
from .services import start_service
```

---

## 5. Best Practices

- ✅ Use `snake_case.py` filenames for modules
- ✅ Group related modules in a package
- ✅ Avoid wildcard imports: `from module import *`
- ✅ Handle circular imports using import inside functions

```python
# Avoiding circular import
# file_a.py

def func_a():
    from file_b import func_b
    func_b()
```

---

## 6. Real-world Example: E-commerce Module

```python
# cart.py
def calculate_total(items):
    return sum(price for name, price in items)

# main.py
from cart import calculate_total
items = [("Laptop", 1000), ("Mouse", 50)]
print(f"Total: ${calculate_total(items)}")
```

---

## 7. Hands-on Practice

### Exercise 1: Create Custom Module

- Create `area.py`
- Define `area_of_circle(radius)`
- Import and use it from another file

### Exercise 2: Build Your Own Package

- Create folder `shapes`
- Add `__init__.py`, `circle.py`, `rectangle.py`
- Implement shape functions and use them in `main.py`

### Exercise 3: Explore Built-in Modules

- Use `random` to generate lottery numbers
- Use `math` to calculate square roots and powers

---

## 🔄 Summary

Today you learned:

- What modules and packages are
- How to organize and import code
- Use built-in and third-party libraries
- Apply real-world modularization

---

## 🔍 Common Issues and Fixes

| Issue                  | Solution                                |
| ---------------------- | --------------------------------------- |
| `ModuleNotFoundError`  | Check file path and Python path         |
| `ImportError`          | Use correct syntax for relative imports |
| `Circular ImportError` | Move imports inside functions           |

## 📚 Additional Resources

- [Python Modules – W3Schools](https://www.w3schools.com/python/python_modules.asp)
- [Python Packages – Real Python](https://realpython.com/python-modules-packages/)
- [PEP 8 – Import Guidelines](https://peps.python.org/pep-0008/#imports)

## ✅ Knowledge Check

1. What is the difference between a module and a package?
2. How can you import a function from a different file?
3. What does `__init__.py` do?
4. When should you use relative imports?
5. What are some best practices for avoiding import errors?

> **Navigation**
>
> - [← Day 3: Python Functions](03-Python-Functions.md)
> - [Day 5: Python File I/O →](05-Python-File-Handling.md)

