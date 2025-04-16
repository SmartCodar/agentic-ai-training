# Day 4 - Python Modules and Packages: Code Organization and Reuse

## Overview
This lesson explores Python modules and packages, teaching you how to organize, import, and share code effectively. You'll learn how to use Python's module system to create maintainable and scalable applications while leveraging the vast ecosystem of third-party packages.

## Learning Objectives
- Understand Python modules and their importance
- Learn to create and import modules
- Master package management with pip
- Work with virtual environments
- Use popular third-party packages effectively

## Prerequisites
- Understanding of Python variables and data types
- Knowledge of flow control (if statements, loops)
- Understanding of functions and their usage
- Python 3.x installed on your computer
- Basic command line knowledge

## Time Estimate
- Reading: 35 minutes
- Practice: 50 minutes
- Assignments: 45 minutes

---

## 1. Understanding Modules

### What is a Module?
- A module is a Python file containing code
- Modules help organize related code together
- They provide a way to reuse code across different projects

### Creating a Module
```python
# math_operations.py
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# Constants
PI = 3.14159
E = 2.71828
```

### Importing Modules
```python
# Different ways to import
import math_operations                      # Import entire module
from math_operations import add, subtract   # Import specific functions
from math_operations import *              # Import all (not recommended)
import math_operations as mo               # Import with alias

# Using imported functions
result1 = math_operations.add(5, 3)
result2 = add(10, 2)                      # When imported directly
result3 = mo.multiply(4, 6)               # Using alias
```

### Module Search Path
```python
import sys

# View module search paths
for path in sys.path:
    print(path)

# Add a custom path
sys.path.append('/path/to/your/modules')
```

---

## 2. Working with Packages

### Package Structure
```plaintext
my_package/
│
├── __init__.py          # Makes the directory a package
├── module1.py           # Individual modules
├── module2.py
│
└── subpackage/          # Nested package
    ├── __init__.py
    └── module3.py
```

### Creating a Package
```python
# __init__.py
from .module1 import function1
from .module2 import function2

__version__ = '1.0.0'
```

```python
# module1.py
def function1():
    return "This is function1"

# module2.py
def function2():
    return "This is function2"
```

### Installing Packages with pip
```bash
# Install a package
pip install package_name

# Install specific version
pip install package_name==1.2.3

# Install from requirements.txt
pip install -r requirements.txt

# Upgrade a package
pip install --upgrade package_name

# Uninstall a package
pip uninstall package_name
```

### Virtual Environments
```bash
# Create a virtual environment
python -m venv myenv

# Activate virtual environment
# On Windows:
myenv\Scripts\activate
# On Unix or MacOS:
source myenv/bin/activate

# Deactivate virtual environment
deactivate
```

### Managing Dependencies
```bash
# Generate requirements.txt
pip freeze > requirements.txt
```

```plaintext
# requirements.txt example
requests==2.28.1
pandas==1.5.2
numpy==1.23.5
```

### Example: Creating a Project Structure
```plaintext
my_project/
│
├── README.md           # Project documentation
├── requirements.txt    # Project dependencies
├── setup.py           # Package configuration
├── .gitignore         # Git ignore file
│
├── src/               # Source code
│   └── my_package/
│       ├── __init__.py
│       ├── core.py
│       └── utils.py
│
├── tests/             # Test files
│   ├── __init__.py
│   ├── test_core.py
│   └── test_utils.py
│
└── docs/              # Documentation
    ├── installation.md
    └── usage.md
```

---

## 3. Popular Third-Party Packages

### Data Science and Analysis
```python
# NumPy for numerical computations
import numpy as np

# Create an array
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Mean: {np.mean(arr)}")
print(f"Standard deviation: {np.std(arr)}")

# Pandas for data manipulation
import pandas as pd

# Create a DataFrame
data = {
    'Name': ['John', 'Alice', 'Bob'],
    'Age': [28, 24, 32],
    'City': ['New York', 'San Francisco', 'Chicago']
}
df = pd.DataFrame(data)
print("\nDataFrame:")
print(df)
```

### Web Requests and APIs
```python
# Requests for HTTP operations
import requests

# Make a GET request
response = requests.get('https://api.github.com/user', auth=('user', 'pass'))

# Check status and get JSON data
if response.status_code == 200:
    data = response.json()
    print(f"User data: {data}")
```

### File Operations
```python
# Pathlib for file path operations
from pathlib import Path

# Create and manipulate paths
base_dir = Path('project')
data_dir = base_dir / 'data'
output_file = data_dir / 'results.txt'

# Create directories
data_dir.mkdir(parents=True, exist_ok=True)

# Write to file
output_file.write_text('Hello, World!')

# Read from file
content = output_file.read_text()
print(f"File content: {content}")
```

## 4. Real-world Examples

### Example 1: Data Processing Pipeline
```python
# data_pipeline.py
import pandas as pd
from pathlib import Path
from typing import Dict, List

class DataProcessor:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def read_csv_files(self) -> Dict[str, pd.DataFrame]:
        """Read all CSV files from input directory."""
        data_frames = {}
        for file_path in self.input_dir.glob('*.csv'):
            data_frames[file_path.stem] = pd.read_csv(file_path)
        return data_frames
    
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data processing steps."""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.fillna(df.mean(numeric_only=True))
        
        # Add calculated columns
        if 'price' in df.columns and 'quantity' in df.columns:
            df['total'] = df['price'] * df['quantity']
        
        return df
    
    def save_results(self, data_frames: Dict[str, pd.DataFrame]) -> None:
        """Save processed data to output directory."""
        for name, df in data_frames.items():
            output_path = self.output_dir / f"{name}_processed.csv"
            df.to_csv(output_path, index=False)
            print(f"Saved processed data to {output_path}")
    
    def run_pipeline(self) -> None:
        """Execute the complete data processing pipeline."""
        print("Starting data processing pipeline...")
        
        # Read data
        data_frames = self.read_csv_files()
        print(f"Read {len(data_frames)} CSV files")
        
        # Process each dataset
        processed_frames = {
            name: self.process_data(df)
            for name, df in data_frames.items()
        }
        
        # Save results
        self.save_results(processed_frames)
        print("Pipeline completed successfully!")

# Usage example
if __name__ == '__main__':
    processor = DataProcessor('raw_data', 'processed_data')
    processor.run_pipeline()
```

### Example 2: API Client Package
```python
# api_client/__init__.py
from .client import APIClient

__version__ = '1.0.0'
```

```python
# api_client/client.py
import requests
from typing import Dict, Any, Optional

class APIClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}'
            })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to the API."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            raise
    
    def get_user(self, user_id: int) -> Dict[str, Any]:
        """Get user information."""
        return self._make_request('GET', f'users/{user_id}')
    
    def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user."""
        return self._make_request('POST', 'users', json=user_data)
    
    def update_user(self, user_id: int, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user information."""
        return self._make_request('PUT', f'users/{user_id}', json=user_data)
    
    def delete_user(self, user_id: int) -> Dict[str, Any]:
        """Delete a user."""
        return self._make_request('DELETE', f'users/{user_id}')

# Usage example
if __name__ == '__main__':
    client = APIClient('https://api.example.com', api_key='your-api-key')
    
    # Get user info
    user = client.get_user(123)
    print(f"User data: {user}")
    
    # Create new user
    new_user = client.create_user({
        'name': 'John Doe',
        'email': 'john@example.com'
    })
    print(f"Created user: {new_user}")
```

## 5. Best Practices

### Package Organization
- Keep related functionality together
- Use clear and descriptive module names
- Implement proper error handling
- Include comprehensive documentation

### Dependency Management
- Use virtual environments for each project
- Keep dependencies up to date
- Pin dependency versions in requirements.txt
- Document all requirements

### Code Quality
- Follow PEP 8 style guide
- Write unit tests
- Use type hints
- Include docstrings

## 6. Summary

### Key Takeaways
- Modules help organize code into reusable units
- Packages extend modularity to directory level
- Virtual environments isolate project dependencies
- Python has a rich ecosystem of third-party packages

### What's Next
- Object-Oriented Programming in Python
- Testing and Debugging
- Web Development with Python

---

> **Navigation**
> - [← Python Functions](03-Python-Functions-Modular-Programming.md)
> - [Object-Oriented Programming →](05-Python-OOP.md)
