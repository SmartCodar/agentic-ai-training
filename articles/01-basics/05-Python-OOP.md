# Day 5 - Object-Oriented Programming in Python

## Overview
This lesson introduces Object-Oriented Programming (OOP) in Python, teaching you how to create and use classes, objects, and implement key OOP concepts like inheritance and polymorphism. You'll learn how to structure your code using objects to model real-world entities and relationships.

## Learning Objectives
- Understand OOP fundamentals and class creation
- Master inheritance and polymorphism
- Learn encapsulation and abstraction
- Implement class methods and properties
- Work with class relationships

## Prerequisites
- Completion of [Python Basics](01-Python-Basics-Variables-Types-Operators.md)
- Completion of [Flow Control](02-Python-Flow-Control-Loops-Conditions.md)
- Completion of [Functions](03-Python-Functions-Modular-Programming.md)
- Completion of [Modules and Packages](04-Python-Modules-Packages.md)
- Python 3.x installed on your computer

## Time Estimate
- Reading: 40 minutes
- Practice: 50 minutes
- Assignments: 45 minutes

---

## 1. OOP Fundamentals

### What is Object-Oriented Programming?
Object-Oriented Programming (OOP) is a programming paradigm that organizes code into objects, which are instances of classes. Each object contains:
- Data (attributes/properties)
- Code (methods/behaviors)

Think of a class as a blueprint and objects as instances created from that blueprint.

### Key OOP Concepts
1. **Classes**: Templates for creating objects
2. **Objects**: Instances of classes
3. **Attributes**: Data stored in objects
4. **Methods**: Functions that operate on objects

### Classes and Objects Example
```python
# Define a class (blueprint)
class Car:
    # Class attribute (shared by all instances)
    wheels = 4
    
    # Constructor (initializes instance attributes)
    def __init__(self, make: str, model: str, year: int):
        # Instance attributes (unique to each instance)
        self.make = make    # The car's manufacturer
        self.model = model  # The car's model name
        self.year = year    # The car's manufacturing year
        self.speed = 0      # Current speed (starts at 0)
        self._mileage = 0   # Protected attribute for mileage
    
    # Instance method (behavior)
    def accelerate(self, speed_increase: int) -> None:
        """Increase the car's speed.
        
        Args:
            speed_increase: Amount to increase speed by
        """
        self.speed += speed_increase
        print(f"Speed increased to {self.speed} mph")
    
    def brake(self, speed_decrease: int) -> None:
        """Decrease the car's speed safely.
        
        Args:
            speed_decrease: Amount to decrease speed by
        """
        self.speed = max(0, self.speed - speed_decrease)
        print(f"Speed decreased to {self.speed} mph")
    
    def get_info(self) -> str:
        """Return formatted car information.
        
        Returns:
            str: Formatted string with car details
        """
        return f"{self.year} {self.make} {self.model}"
    
    @property
    def mileage(self) -> float:
        """Get the car's mileage (read-only)."""
        return self._mileage

# Creating and using objects (instances)
def demonstrate_car_usage():
    # Create two different car objects
    toyota = Car("Toyota", "Camry", 2023)
    tesla = Car("Tesla", "Model 3", 2024)
    
    # Access class attribute
    print(f"All cars have {Car.wheels} wheels")
    
    # Use instance methods
    print(toyota.get_info())  # Output: 2023 Toyota Camry
    toyota.accelerate(30)     # Output: Speed increased to 30 mph
    toyota.brake(10)          # Output: Speed decreased to 20 mph
    
    # Access instance attributes
    print(f"Tesla model: {tesla.model}")
    print(f"Tesla speed: {tesla.speed} mph")
    
    # Access protected attribute through property
    print(f"Toyota mileage: {toyota.mileage} miles")

# Run the demonstration
demonstrate_car_usage()
```

### Understanding the Code
1. **Class Definition**:
   - `class Car:` creates a new class named Car
   - `wheels = 4` is a class attribute shared by all instances
   - `__init__` is the constructor method

2. **Attributes**:
   - Class attributes: Shared by all instances (e.g., `wheels`)
   - Instance attributes: Unique to each instance (e.g., `make`, `model`)
   - Protected attributes: Prefixed with underscore (e.g., `_mileage`)

3. **Methods**:
   - Instance methods: Operate on instance data (e.g., `accelerate`, `brake`)
   - Properties: Control access to attributes (e.g., `@property mileage`)
   - Each method uses `self` to reference the instance

4. **Object Creation and Usage**:
   - Objects are created by calling the class (e.g., `Car("Toyota", "Camry", 2023)`)
   - Methods are called using dot notation (e.g., `toyota.accelerate(30)`)
   - Attributes are accessed using dot notation (e.g., `tesla.model`)
```python
class Car:
    def __init__(self, make: str, model: str, year: int):
        self.make = make
        self.model = model
        self.year = year
        self.speed = 0
    
    def accelerate(self, speed_increase: int) -> None:
        """Increase the car's speed."""
        self.speed += speed_increase
        print(f"Speed increased to {self.speed} mph")
    
    def brake(self, speed_decrease: int) -> None:
        """Decrease the car's speed."""
        self.speed = max(0, self.speed - speed_decrease)
        print(f"Speed decreased to {self.speed} mph")
    
    def get_info(self) -> str:
        """Return car information."""
        return f"{self.year} {self.make} {self.model}"

# Creating and using objects
my_car = Car("Toyota", "Camry", 2023)
print(my_car.get_info())
my_car.accelerate(30)
my_car.brake(10)
```

### Instance vs. Class Variables: Deep Dive

#### Understanding Variable Scope in Classes
Class variables and instance variables serve different purposes and have different scopes. Here's a detailed example:
```python
class BankAccount:
    # Class variables (shared by all instances)
    interest_rate = 0.02    # Standard interest rate
    bank_name = "MyBank"    # Bank name
    total_accounts = 0      # Counter for all accounts
    
    def __init__(self, account_number: str, balance: float = 0):
        # Instance variables (unique to each account)
        self.account_number = account_number  # Account identifier
        self.balance = balance                # Current balance
        self._transactions = []               # Transaction history
        
        # Update class-level counter
        BankAccount.total_accounts += 1
    
    @classmethod
    def change_interest_rate(cls, new_rate: float) -> None:
        """Change interest rate for all accounts.
        
        Args:
            new_rate: New interest rate (e.g., 0.03 for 3%)
        """
        if 0 <= new_rate <= 0.1:  # Validate rate (0-10%)
            cls.interest_rate = new_rate
            print(f"Interest rate changed to {new_rate:.1%}")
        else:
            raise ValueError("Interest rate must be between 0% and 10%")
    
    def apply_interest(self) -> None:
        """Apply current interest rate to balance."""
        interest = self.balance * self.interest_rate
        self.balance += interest
        
        # Record transaction
        self._transactions.append({
            'type': 'interest',
            'amount': interest,
            'balance': self.balance
        })
        
        print(f"Applied interest: ${interest:.2f}")
        print(f"New balance: ${self.balance:.2f}")
    
    def get_statement(self) -> None:
        """Print account statement with transactions."""
        print(f"\n{self.bank_name} - Account Statement")
        print(f"Account: {self.account_number}")
        print("-" * 40)
        
        for transaction in self._transactions:
            print(f"Type: {transaction['type']}")
            print(f"Amount: ${transaction['amount']:.2f}")
            print(f"Balance: ${transaction['balance']:.2f}")
            print("-" * 40)

# Demonstrate class and instance variables
def demonstrate_bank_accounts():
    # Create accounts
    print(f"Bank: {BankAccount.bank_name}")
    print(f"Initial interest rate: {BankAccount.interest_rate:.1%}\n")
    
    # Create two accounts
    account1 = BankAccount("1234", 1000)
    account2 = BankAccount("5678", 2000)
    
    print(f"Total accounts created: {BankAccount.total_accounts}\n")
    
    # Apply interest to both accounts
    account1.apply_interest()
    account2.apply_interest()
    
    # Change interest rate for all accounts
    BankAccount.change_interest_rate(0.03)
    
    # Apply new interest rate
    account1.apply_interest()
    account2.apply_interest()
    
    # Show statement for account1
    account1.get_statement()

# Run the demonstration
demonstrate_bank_accounts()
```

### Key Points About Variable Scope

1. **Class Variables**:
   - Shared by all instances of the class
   - Defined directly in the class (outside methods)
   - Accessed via `ClassName.variable` or `self.variable`
   - Common uses:
     - Constants (e.g., `bank_name`)
     - Shared state (e.g., `interest_rate`)
     - Counters (e.g., `total_accounts`)

2. **Instance Variables**:
   - Unique to each instance
   - Defined in `__init__` or instance methods
   - Always accessed via `self.variable`
   - Common uses:
     - Object state (e.g., `balance`)
     - Instance-specific data (e.g., `account_number`)
     - Private data (e.g., `_transactions`)

3. **Method Types**:
   - Instance methods: Regular methods that access instance data
   - Class methods: Methods that access class data (use `@classmethod`)
   - Static methods: Utility methods that don't access instance or class data

4. **Best Practices**:
   - Use class variables for data shared across all instances
   - Use instance variables for instance-specific data
   - Use class methods to modify class variables
   - Protect instance data with underscore prefix when needed

## 2. Inheritance and Polymorphism

### Understanding Inheritance
Inheritance is a mechanism that allows a class to inherit attributes and methods from another class. This promotes code reuse and establishes relationships between classes.

#### Key Concepts:
1. **Base Class (Parent)**: The class being inherited from
2. **Derived Class (Child)**: The class that inherits
3. **super()**: Function to call methods from parent class
4. **Method Overriding**: Redefining methods from parent class

### Types of Inheritance
1. **Single Inheritance**: Class inherits from one base class
2. **Multiple Inheritance**: Class inherits from multiple base classes
3. **Multilevel Inheritance**: Class inherits from a derived class
4. **Hierarchical Inheritance**: Multiple classes inherit from one base class

### Real-world Example: Employee Management System

```python
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

# Base employee class
class Employee(ABC):
    # Class variable to track all employees
    total_employees = 0
    
    def __init__(self, name: str, id: str, salary: float):
        self.name = name
        self.id = id
        self._salary = salary  # Protected attribute
        self._hire_date = datetime.now()
        self._projects: List[str] = []  # Project assignments
        
        # Increment employee counter
        Employee.total_employees += 1
    
    @abstractmethod
    def calculate_bonus(self) -> float:
        """Calculate employee's annual bonus."""
        pass
    
    @property
    def salary(self) -> float:
        """Get employee's salary (read-only)."""
        return self._salary
    
    def assign_project(self, project_name: str) -> None:
        """Assign employee to a project."""
        self._projects.append(project_name)
        print(f"{self.name} assigned to {project_name}")
    
    def get_info(self) -> str:
        """Get employee information."""
        return f"Employee: {self.name} (ID: {self.id})"

# Developer class (inherits from Employee)
class Developer(Employee):
    def __init__(self, name: str, id: str, salary: float, 
                 programming_languages: List[str]):
        super().__init__(name, id, salary)
        self.programming_languages = programming_languages
        self._bugs_fixed = 0
    
    def calculate_bonus(self) -> float:
        """Calculate bonus based on bugs fixed and projects."""
        project_bonus = len(self._projects) * 1000
        bug_bonus = self._bugs_fixed * 50
        return project_bonus + bug_bonus
    
    def fix_bug(self) -> None:
        """Record a bug fix."""
        self._bugs_fixed += 1
        print(f"{self.name} fixed a bug! Total fixes: {self._bugs_fixed}")
    
    def get_info(self) -> str:
        """Override parent method with developer-specific info."""
        base_info = super().get_info()
        return f"{base_info}\nLanguages: {', '.join(self.programming_languages)}"

# Manager class (inherits from Employee)
class Manager(Employee):
    def __init__(self, name: str, id: str, salary: float, 
                 department: str):
        super().__init__(name, id, salary)
        self.department = department
        self._team: List[Employee] = []
    
    def calculate_bonus(self) -> float:
        """Calculate bonus based on team size and projects."""
        team_bonus = len(self._team) * 2000
        project_bonus = len(self._projects) * 1500
        return team_bonus + project_bonus
    
    def add_team_member(self, employee: Employee) -> None:
        """Add an employee to the team."""
        self._team.append(employee)
        print(f"{employee.name} added to {self.name}'s team")
    
    def get_team_info(self) -> str:
        """Get information about the team."""
        team_info = [member.get_info() for member in self._team]
        return f"\nTeam Members:\n" + "\n".join(team_info)

# Demonstrate employee management system
def demonstrate_employee_system():
    # Create employees
    dev1 = Developer("Alice Smith", "D001", 75000, 
                    ["Python", "JavaScript"])
    dev2 = Developer("Bob Johnson", "D002", 70000, 
                    ["Java", "C++"])
    manager = Manager("Carol Williams", "M001", 95000, 
                     "Development")
    
    # Assign projects
    dev1.assign_project("Website Redesign")
    dev2.assign_project("Mobile App")
    manager.assign_project("Digital Transformation")
    
    # Record bug fixes
    dev1.fix_bug()
    dev1.fix_bug()
    dev2.fix_bug()
    
    # Build team
    manager.add_team_member(dev1)
    manager.add_team_member(dev2)
    
    # Print information
    print(f"\nTotal Employees: {Employee.total_employees}")
    print(f"\nManager Information:")
    print(manager.get_info())
    print(manager.get_team_info())
    
    print(f"\nBonuses:")
    print(f"{dev1.name}: ${dev1.calculate_bonus():,.2f}")
    print(f"{dev2.name}: ${dev2.calculate_bonus():,.2f}")
    print(f"{manager.name}: ${manager.calculate_bonus():,.2f}")

# Run the demonstration
demonstrate_employee_system()
```

### Understanding the Employee Management System

1. **Abstract Base Class (Employee)**:
   - Defines common interface for all employees
   - Uses `@abstractmethod` for bonus calculation
   - Tracks total employees with class variable
   - Provides shared functionality (projects, info)

2. **Developer Class**:
   - Inherits from Employee
   - Adds programming languages and bug tracking
   - Overrides bonus calculation based on performance
   - Extends employee info with languages

3. **Manager Class**:
   - Inherits from Employee
   - Adds team management functionality
   - Calculates bonus based on team and projects
   - Provides team overview capabilities

4. **Inheritance Features Demonstrated**:
   - Method overriding (`get_info`, `calculate_bonus`)
   - Protected attributes (`_salary`, `_bugs_fixed`)
   - Abstract methods enforcing interface
   - Property decorators for encapsulation

### Shape Hierarchy Example
```python
from abc import ABC, abstractmethod
from math import pi

# Abstract base class
class Shape(ABC):
    def __init__(self, color: str):
        self.color = color
    
    @abstractmethod
    def area(self) -> float:
        """Calculate shape area."""
        pass
    
    @abstractmethod
    def perimeter(self) -> float:
        """Calculate shape perimeter."""
        pass
    
    def describe(self) -> str:
        """Return shape description."""
        return f"A {self.color} shape"

# Derived class: Rectangle
class Rectangle(Shape):
    def __init__(self, width: float, height: float, color: str):
        super().__init__(color)  # Call parent constructor
        self.width = width
        self.height = height
    
    def area(self) -> float:
        return self.width * self.height
    
    def perimeter(self) -> float:
        return 2 * (self.width + self.height)
    
    def describe(self) -> str:
        # Override parent method
        return f"A {self.color} rectangle with width {self.width} and height {self.height}"

# Derived class: Circle
class Circle(Shape):
    def __init__(self, radius: float, color: str):
        super().__init__(color)
        self.radius = radius
    
    def area(self) -> float:
        return pi * self.radius ** 2
    
    def perimeter(self) -> float:
        return 2 * pi * self.radius
    
    def describe(self) -> str:
        return f"A {self.color} circle with radius {self.radius}"

# Demonstrate inheritance and polymorphism
def demonstrate_shapes():
    # Create shape objects
    rect = Rectangle(5, 3, "blue")
    circle = Circle(2, "red")
    
    # Store shapes in a list (polymorphism)
    shapes = [rect, circle]
    
    # Process all shapes uniformly
    for shape in shapes:
        print(f"\n{shape.describe()}")
        print(f"Area: {shape.area():.2f}")
        print(f"Perimeter: {shape.perimeter():.2f}")

# Run the demonstration
demonstrate_shapes()
```

### Understanding the Inheritance Example

1. **Abstract Base Class (Shape)**:
   - Defines the common interface for all shapes
   - Uses `@abstractmethod` to enforce implementation
   - Provides some shared functionality (`describe`)

2. **Concrete Classes (Rectangle, Circle)**:
   - Inherit from Shape using `class ClassName(Shape)`
   - Must implement all abstract methods
   - Can override parent methods
   - Can add new methods and attributes

3. **Method Implementation**:
   - `super().__init__()` calls parent constructor
   - Abstract methods must be implemented
   - Parent methods can be extended or overridden

4. **Polymorphism in Action**:
   - Different shapes stored in same list
   - Each shape handles area/perimeter differently
   - Code works with any Shape subclass

### Real-world Application: Employee Management
```python
class Animal:
    def __init__(self, name: str, species: str):
        self.name = name
        self.species = species
    
    def make_sound(self) -> str:
        return "Some generic sound"

class Dog(Animal):
    def __init__(self, name: str, breed: str):
        super().__init__(name, species="Dog")
        self.breed = breed
    
    def make_sound(self) -> str:
        return "Woof!"
    
    def fetch(self) -> str:
        return f"{self.name} is fetching the ball"

class Cat(Animal):
    def __init__(self, name: str, indoor: bool):
        super().__init__(name, species="Cat")
        self.indoor = indoor
    
    def make_sound(self) -> str:
        return "Meow!"
    
    def scratch(self) -> str:
        return f"{self.name} is scratching"

# Using inheritance and polymorphism
dog = Dog("Buddy", "Golden Retriever")
cat = Cat("Whiskers", True)

animals = [dog, cat]
for animal in animals:
    print(f"{animal.name} says: {animal.make_sound()}")
```

### Multiple Inheritance
```python
class ElectricSystem:
    def __init__(self, voltage: int):
        self.voltage = voltage
    
    def check_battery(self) -> str:
        return f"Battery voltage: {self.voltage}V"

class Computer:
    def __init__(self, cpu: str, ram: int):
        self.cpu = cpu
        self.ram = ram
    
    def get_specs(self) -> str:
        return f"CPU: {self.cpu}, RAM: {self.ram}GB"

class Laptop(Computer, ElectricSystem):
    def __init__(self, cpu: str, ram: int, voltage: int, battery_life: int):
        Computer.__init__(self, cpu, ram)
        ElectricSystem.__init__(self, voltage)
        self.battery_life = battery_life
    
    def get_info(self) -> str:
        return f"{self.get_specs()}, {self.check_battery()}, Battery Life: {self.battery_life}h"

# Using multiple inheritance
laptop = Laptop("Intel i7", 16, 12, 8)
print(laptop.get_info())
```

## 3. Encapsulation and Abstraction

### Private and Protected Members
```python
class Employee:
    def __init__(self, name: str, salary: float):
        self.name = name           # Public
        self._department = None    # Protected
        self.__salary = salary     # Private
    
    def get_salary(self) -> float:
        """Get employee salary."""
        return self.__salary
    
    def set_department(self, department: str) -> None:
        """Set employee department."""
        self._department = department
    
    def get_info(self) -> str:
        """Get employee information."""
        return f"Name: {self.name}, Department: {self._department}, Salary: ${self.__salary:,.2f}"

# Using encapsulation
employee = Employee("John Doe", 75000)
employee.set_department("Engineering")
print(employee.get_info())
# print(employee.__salary)  # This would raise an error
```

### Properties and Decorators
```python
class Temperature:
    def __init__(self, celsius: float = 0):
        self._celsius = celsius
    
    @property
    def celsius(self) -> float:
        """Get temperature in Celsius."""
        return self._celsius
    
    @celsius.setter
    def celsius(self, value: float) -> None:
        """Set temperature in Celsius."""
        if value < -273.15:
            raise ValueError("Temperature below absolute zero!")
        self._celsius = value
    
    @property
    def fahrenheit(self) -> float:
        """Get temperature in Fahrenheit."""
        return (self.celsius * 9/5) + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value: float) -> None:
        """Set temperature in Fahrenheit."""
        self.celsius = (value - 32) * 5/9

# Using properties
temp = Temperature(25)
print(f"Celsius: {temp.celsius}°C")
print(f"Fahrenheit: {temp.fahrenheit}°F")

temp.fahrenheit = 100
print(f"New Celsius: {temp.celsius}°C")
```

## 4. Advanced OOP Concepts

### Abstract Base Classes
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        """Calculate shape area."""
        pass
    
    @abstractmethod
    def perimeter(self) -> float:
        """Calculate shape perimeter."""
        pass

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    
    def area(self) -> float:
        return self.width * self.height
    
    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius
    
    def area(self) -> float:
        return 3.14159 * self.radius ** 2
    
    def perimeter(self) -> float:
        return 2 * 3.14159 * self.radius

# Using abstract classes
shapes = [Rectangle(5, 3), Circle(2)]
for shape in shapes:
    print(f"Area: {shape.area():.2f}")
    print(f"Perimeter: {shape.perimeter():.2f}")
```

### Class Methods and Static Methods
```python
class DateUtil:
    @staticmethod
    def is_leap_year(year: int) -> bool:
        """Check if a year is a leap year."""
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    
    @classmethod
    def create_date(cls, date_str: str):
        """Create a date object from string (YYYY-MM-DD)."""
        year, month, day = map(int, date_str.split('-'))
        return cls(year, month, day)
    
    def __init__(self, year: int, month: int, day: int):
        self.year = year
        self.month = month
        self.day = day
    
    def __str__(self) -> str:
        return f"{self.year}-{self.month:02d}-{self.day:02d}"

# Using class and static methods
print(f"2024 is leap year: {DateUtil.is_leap_year(2024)}")
date = DateUtil.create_date("2025-04-15")
print(f"Created date: {date}")
```

## 5. Best Practices

### SOLID Principles
1. **Single Responsibility**: Each class should have one job
2. **Open/Closed**: Open for extension, closed for modification
3. **Liskov Substitution**: Derived classes must be substitutable for base classes
4. **Interface Segregation**: Many specific interfaces better than one general
5. **Dependency Inversion**: Depend on abstractions, not concretions

### Design Patterns
```python
# Singleton Pattern Example
class Configuration:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.settings = {}
        return cls._instance
    
    def set_setting(self, key: str, value: str) -> None:
        self.settings[key] = value
    
    def get_setting(self, key: str) -> str:
        return self.settings.get(key)

# Using Singleton
config1 = Configuration()
config1.set_setting("theme", "dark")

config2 = Configuration()
print(config2.get_setting("theme"))  # Will print "dark"
print(config1 is config2)  # Will print True
```

## 6. Real-world Example

### E-commerce System
```python
from abc import ABC, abstractmethod
from typing import List, Dict
from datetime import datetime

class Product:
    def __init__(self, id: int, name: str, price: float):
        self.id = id
        self.name = name
        self.price = price
    
    def __str__(self) -> str:
        return f"{self.name} (${self.price:.2f})"

class Discount(ABC):
    @abstractmethod
    def apply(self, price: float) -> float:
        pass

class PercentageDiscount(Discount):
    def __init__(self, percentage: float):
        self.percentage = percentage
    
    def apply(self, price: float) -> float:
        return price * (1 - self.percentage / 100)

class FlatDiscount(Discount):
    def __init__(self, amount: float):
        self.amount = amount
    
    def apply(self, price: float) -> float:
        return max(0, price - self.amount)

class CartItem:
    def __init__(self, product: Product, quantity: int):
        self.product = product
        self.quantity = quantity
    
    def get_total(self) -> float:
        return self.product.price * self.quantity

class ShoppingCart:
    def __init__(self):
        self.items: List[CartItem] = []
        self.discount: Discount = None
    
    def add_item(self, product: Product, quantity: int = 1) -> None:
        self.items.append(CartItem(product, quantity))
    
    def set_discount(self, discount: Discount) -> None:
        self.discount = discount
    
    def get_subtotal(self) -> float:
        return sum(item.get_total() for item in self.items)
    
    def get_total(self) -> float:
        subtotal = self.get_subtotal()
        if self.discount:
            return self.discount.apply(subtotal)
        return subtotal

class Order:
    def __init__(self, cart: ShoppingCart, customer_email: str):
        self.cart = cart
        self.customer_email = customer_email
        self.order_date = datetime.now()
        self.status = "Pending"
    
    def process(self) -> None:
        # Simulate order processing
        print(f"Processing order for {self.customer_email}")
        print(f"Total amount: ${self.cart.get_total():.2f}")
        self.status = "Processed"

# Using the e-commerce system
def test_ecommerce_system():
    # Create products
    laptop = Product(1, "Laptop", 999.99)
    mouse = Product(2, "Mouse", 29.99)
    keyboard = Product(3, "Keyboard", 59.99)
    
    # Create shopping cart and add items
    cart = ShoppingCart()
    cart.add_item(laptop)
    cart.add_item(mouse, 2)
    cart.add_item(keyboard)
    
    # Apply percentage discount
    cart.set_discount(PercentageDiscount(10))
    
    # Create and process order
    order = Order(cart, "customer@example.com")
    order.process()

# Run the test
test_ecommerce_system()
```

## 7. Summary

### Key Takeaways
- OOP helps organize code into reusable, maintainable classes
- Inheritance enables code reuse and hierarchical relationships
- Encapsulation protects data and implementation details
- Polymorphism allows flexible behavior through interfaces
- Properties and decorators enhance class functionality

### What's Next
- Testing and Debugging Python Code
- Web Development with Python
- Working with Databases

---

> **Navigation**
> - [← Python Modules and Packages](04-Python-Modules-Packages.md)
> - [Testing and Debugging →](06-Python-Testing-Debugging.md)
