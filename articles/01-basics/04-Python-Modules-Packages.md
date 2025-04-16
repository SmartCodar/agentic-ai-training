# Day 4: Python Modules and Packages

## 1. Understanding Modules 🧩
A module is a Python file containing reusable code. Let's create our first module:

## 1.1 Creating Your First Module
```python
# math_utils.py

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

# Constants
PI = 3.14159
GOLDEN_RATIO = 1.61803
```

## 1.2 Using Modules
```python
# main.py

# Import entire module
import math_utils

# Import specific items
from math_utils import add, PI

# Import with alias
import math_utils as mu

# Usage examples
result1 = math_utils.multiply(5, 3)  # Using full module name
result2 = add(10, 20)                # Using imported function
result3 = mu.multiply(4, 2)          # Using alias
```

## 2. Creating a Package 📦
A package is a directory containing modules. Let's create a data analysis package:

```plaintext
data_analysis/
├── __init__.py           # Makes it a package
├── readers/              # Subpackage for data readers
│   ├── __init__.py
│   ├── csv_reader.py
│   └── json_reader.py
├── processors/           # Subpackage for data processing
│   ├── __init__.py
│   ├── cleaner.py
│   └── transformer.py
└── utils/                # Utility functions
    ├── __init__.py
    └── helpers.py
```

### 2.1 Package Implementation

```python
# data_analysis/readers/csv_reader.py
from typing import List, Dict
import csv

def read_csv(filepath: str) -> List[Dict]:
    """Read data from CSV file."""
    with open(filepath, 'r') as file:
        return list(csv.DictReader(file))

# data_analysis/processors/cleaner.py
def remove_nulls(data: List[Dict]) -> List[Dict]:
    """Remove rows with null values."""
    return [row for row in data if all(row.values())]

# data_analysis/utils/helpers.py
from datetime import datetime

def get_timestamp() -> str:
    """Get current timestamp."""
    return datetime.now().isoformat()
```

### 2.2 Package Configuration
```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="data_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas>=1.5.0',
        'numpy>=1.20.0'
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A data analysis toolkit",
    python_requires=">=3.11"
)
```

## 3. Using the Package
```python
# Example usage
from data_analysis.readers.csv_reader import read_csv
from data_analysis.processors.cleaner import remove_nulls
from data_analysis.utils.helpers import get_timestamp

# Read and process data
data = read_csv('data.csv')
clean_data = remove_nulls(data)
print(f"Processing completed at: {get_timestamp()}")
```

## 4. Best Practices 📖

1. **Module Organization**
   - One class/purpose per module
   - Related modules in packages
   - Clear, descriptive names

2. **Import Guidelines**
   - Import standard library first
   - Then third-party packages
   - Finally, local modules
   - Use absolute imports

3. **Package Structure**
   - Keep `__init__.py` simple
   - Use meaningful directory names
   - Include README and documentation
   - Add type hints

4. **Distribution**
   - Use `setup.py` or `pyproject.toml`
   - Include requirements
   - Add version control
   - Write tests

## 5. Practice Exercise 🎯

Create a package called `text_processor` that can:
1. Load text files
2. Count words and characters
3. Find common phrases
4. Generate statistics

```plaintext
text_processor/
├── __init__.py
├── loader.py      # File loading functions
├── analyzer.py    # Text analysis tools
├── stats.py       # Statistical functions
└── utils.py       # Helper utilities
```

Implement the basic structure and add functionality step by step.

## 6. Common Issues and Solutions 🔧

1. **Import Errors**
   ```python
   # Wrong
   from ..module import function  # Relative import error
   
   # Right
   from package.module import function  # Absolute import
   ```

2. **Circular Imports**
   ```python
   # Avoid this pattern
   # a.py
   from b import function_b
   
   # b.py
   from a import function_a
   
   # Solution: Restructure or use import in function
   ```

3. **Package Not Found**
   ```bash
   # Install in development mode
   pip install -e .
   
   # Add to PYTHONPATH
   export PYTHONPATH="$PYTHONPATH:/path/to/package"
   ```

## 1. Module Fundamentals

### 1.1 Creating Modules
```python
# calculator.py
from typing import Union, Dict, Any
from dataclasses import dataclass
from enum import Enum, auto

class Operation(Enum):
    """Available calculator operations."""
    ADD = auto()
    SUBTRACT = auto()
    MULTIPLY = auto()
    DIVIDE = auto()

@dataclass
class CalculationResult:
    """Store calculation result and metadata."""
    operation: Operation
    result: float
    inputs: tuple[float, float]
    error: str = ""

class Calculator:
    """Basic calculator implementation."""
    def __init__(self):
        self.history: list[CalculationResult] = []
    
    def calculate(self, a: float, b: float, operation: Operation) -> CalculationResult:
        """Perform calculation and store in history."""
        try:
            if operation == Operation.ADD:
                result = a + b
            elif operation == Operation.SUBTRACT:
                result = a - b
            elif operation == Operation.MULTIPLY:
                result = a * b
            elif operation == Operation.DIVIDE:
                if b == 0:
                    raise ValueError("Division by zero")
                result = a / b
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            calc_result = CalculationResult(operation, result, (a, b))
            self.history.append(calc_result)
            return calc_result
        
        except Exception as e:
            calc_result = CalculationResult(
                operation=operation,
                result=0.0,
                inputs=(a, b),
                error=str(e)
            )
            self.history.append(calc_result)
            return calc_result
    
    def get_history(self) -> list[Dict[str, Any]]:
        """Get calculation history in serializable format."""
        return [
            {
                'operation': calc.operation.name,
                'result': calc.result,
                'inputs': calc.inputs,
                'error': calc.error
            }
            for calc in self.history
        ]

# Only run if module is executed directly
if __name__ == '__main__':
    # Module self-test
    calc = Calculator()
    print(calc.calculate(10, 5, Operation.ADD))
    print(calc.calculate(10, 2, Operation.DIVIDE))
    print(calc.get_history())
```
### 1.2 Importing and Using Modules
```python
# main.py
from typing import List
from calculator import Calculator, Operation

def run_calculations(nums: List[float]) -> None:
    """Run a series of calculations using our calculator module."""
    calc = Calculator()
    
    # Perform operations between consecutive numbers
    for i in range(0, len(nums) - 1, 2):
        a, b = nums[i], nums[i + 1]
        
        # Try all operations
        for op in Operation:
            result = calc.calculate(a, b, op)
            if result.error:
                print(f"Error: {result.error}")
            else:
                print(f"{a} {op.name} {b} = {result.result}")
    
    # Print calculation history
    print("\nCalculation History:")
    for entry in calc.get_history():
        print(f"{entry['operation']}: {entry['inputs']} = {entry['result']}")

# Example usage
if __name__ == '__main__':
    numbers = [10, 5, 15, 3, 20, 4]
    run_calculations(numbers)
```

### 1.3 Module Import Patterns
```python
# Different import styles
from calculator import Calculator, Operation  # Import specific names
import calculator                           # Import whole module
from calculator import *                    # Import all (not recommended)

# Using alias
import calculator as calc
my_calc = calc.Calculator()

# Importing from nested modules
from mypackage.utils.math import Calculator
from mypackage.constants import PI, E

# Relative imports (within a package)
from .calculator import Calculator          # Same directory
from ..utils import format_number           # Parent directory
from .validators import validate_input      # Same directory
```

### 1.4 Module Structure Best Practices
```python
# config.py - Configuration module example
from typing import Dict, Any
from pathlib import Path
import json

class Config:
    """Application configuration manager."""
    def __init__(self, config_path: Path):
        self._config_path = config_path
        self._config: Dict[str, Any] = {}
        self.load()
    
    def load(self) -> None:
        """Load configuration from file."""
        if self._config_path.exists():
            with open(self._config_path) as f:
                self._config = json.load(f)
    
    def save(self) -> None:
        """Save configuration to file."""
        with open(self._config_path, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value
        self.save()

# Module constants
DEFAULT_CONFIG = Path('config.json')
VERSION = '1.0.0'

# Module initialization
def init_config(config_path: Path = DEFAULT_CONFIG) -> Config:
    """Initialize configuration."""
    return Config(config_path)

# Module cleanup
def cleanup() -> None:
    """Cleanup module resources."""
    # Cleanup code here
    pass

# Only run if module is executed directly
if __name__ == '__main__':
    # Module self-test
    config = init_config()
    config.set('app_name', 'MyApp')
    print(config.get('app_name'))
```
## 2. Package Organization

### 2.1 Package Structure
```plaintext
mypackage/
├── __init__.py           # Package initialization
├── calculator.py         # Core calculator module
├── utils/
│   ├── __init__.py      # Utils subpackage
│   ├── validators.py     # Input validation
│   └── formatters.py     # Output formatting
├── config/
│   ├── __init__.py      # Config subpackage
│   ├── settings.py       # Settings management
│   └── defaults.json     # Default configuration
└── tests/
    ├── __init__.py      # Tests package
    ├── test_calculator.py
    └── test_utils.py
```

### 2.2 Package Initialization
```python
# mypackage/__init__.py
from typing import List, Dict, Any
from pathlib import Path

# Version and metadata
__version__ = '1.0.0'
__author__ = 'Your Name'

# Import main components for easier access
from .calculator import Calculator, Operation, CalculationResult
from .utils.validators import validate_input
from .utils.formatters import format_result

# Package initialization
def initialize(config_path: Path = None) -> Dict[str, Any]:
    """Initialize the package with optional configuration."""
    from .config.settings import load_config
    
    config = load_config(config_path) if config_path else {}
    return {
        'version': __version__,
        'config': config
    }

# Cleanup function
def cleanup() -> None:
    """Cleanup package resources."""
    from .config.settings import save_config
    save_config()
```

### 2.3 Subpackage Organization
```python
# mypackage/utils/validators.py
from typing import Any, Union, List
from decimal import Decimal

def validate_number(value: Any) -> Union[int, float, Decimal]:
    """Validate and convert numeric input."""
    if isinstance(value, (int, float, Decimal)):
        return value
    try:
        # Try integer first
        return int(value)
    except ValueError:
        try:
            # Then try float
            return float(value)
        except ValueError:
            try:
                # Finally try Decimal
                return Decimal(str(value))
            except:
                raise ValueError(f"Cannot convert {value} to number")

def validate_operation(op_name: str) -> str:
    """Validate operation name."""
    valid_ops = {'add', 'subtract', 'multiply', 'divide'}
    op_name = op_name.lower()
    if op_name not in valid_ops:
        raise ValueError(f"Invalid operation. Must be one of: {', '.join(valid_ops)}")
    return op_name

# mypackage/utils/formatters.py
from typing import Any
from decimal import Decimal

def format_number(value: Union[int, float, Decimal], precision: int = 2) -> str:
    """Format number with specified precision."""
    if isinstance(value, Decimal):
        return str(value.normalize())
    return f"{value:.{precision}f}".rstrip('0').rstrip('.')

def format_result(operation: str, a: Any, b: Any, result: Any) -> str:
    """Format calculation result."""
    return f"{format_number(a)} {operation} {format_number(b)} = {format_number(result)}"
```

## Importing Modules
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

## 📚 Additional Resources
- [Python Modules Documentation](https://docs.python.org/3/tutorial/modules.html)
- [Python Packages Documentation](https://docs.python.org/3/tutorial/modules.html#packages)
- [PEP 420 – Implicit Namespace Packages](https://peps.python.org/pep-0420/)
- [Real Python: Python Modules and Packages](https://realpython.com/python-modules-packages/)

## ✅ Knowledge Check
1. What is the difference between a module and a package?
2. When should you use relative vs absolute imports?
3. What is the purpose of `__init__.py` files?
4. How do you handle circular imports?
5. What are namespace packages and when should you use them?

## 🔍 Common Issues and Solutions
| Issue | Solution |
|-------|----------|
| Circular Imports | Restructure code or move imports inside functions |
| Module Not Found | Check PYTHONPATH and package structure |
| Import * Issues | Use explicit imports instead |
| Package Conflicts | Use virtual environments |

## 📝 Summary
- Modules help organize related code into reusable units
- Packages provide hierarchical organization of modules
- Proper structuring improves maintainability
- Best practices focus on clarity and organization
- Modern Python features enhance module system usage

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
- Packages provide structured code organization
- Virtual environments manage project dependencies
- Best practices for module and package design
- Common issues and their solutions

## 📚 Additional Resources
- [Python Modules Documentation](https://docs.python.org/3/tutorial/modules.html)
- [Python Packaging Guide](https://packaging.python.org/guides/)
- [Real Python - Python Modules and Packages](https://realpython.com/python-modules-packages/)

---

> **Navigation**
> - [← Python Functions and Modular Programming](03-Python-Functions-Modular-Programming.md)
> - [Python Object-Oriented Programming →](05-Python-OOP.md)
