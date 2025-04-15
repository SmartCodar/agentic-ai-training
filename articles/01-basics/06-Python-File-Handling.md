# Day 6 - File Handling in Python

## Overview
This lesson covers file handling operations in Python, including working with text files, JSON data, CSV files, and binary files. You'll learn how to read, write, and manipulate different file formats efficiently and safely.

## Learning Objectives
- Master file operations (read, write, append)
- Work with different file formats (text, JSON, CSV)
- Handle file paths and directories
- Implement error handling for file operations
- Use context managers (`with` statement)

## Prerequisites
- Completion of [Python Basics](01-Python-Basics-Variables-Types-Operators.md)
- Completion of [Flow Control](02-Python-Flow-Control-Loops-Conditions.md)
- Completion of [Functions](03-Python-Functions-Modular-Programming.md)
- Completion of [Modules and Packages](04-Python-Modules-Packages.md)
- Completion of [Object-Oriented Programming](05-Python-OOP.md)
- Python 3.x installed on your computer

## Time Estimate
- Reading: 35 minutes
- Practice: 45 minutes
- Assignments: 40 minutes

---

## 1. Text File Operations

### Basic File Operations
```python
def demonstrate_basic_file_ops():
    # Writing to a file
    with open('example.txt', 'w') as file:
        file.write('Hello, World!\n')
        file.write('This is a new line.\n')
        
    # Reading entire file
    with open('example.txt', 'r') as file:
        content = file.read()
        print("Entire file content:")
        print(content)
    
    # Reading line by line
    with open('example.txt', 'r') as file:
        print("\nLine by line:")
        for line in file:
            print(f"Line: {line.strip()}")
    
    # Appending to file
    with open('example.txt', 'a') as file:
        file.write('Appended line!\n')
    
    # Reading specific number of characters
    with open('example.txt', 'r') as file:
        print("\nFirst 10 characters:")
        print(file.read(10))

# Run the demonstration
demonstrate_basic_file_ops()
```

### File Modes and Error Handling
```python
def demonstrate_file_modes():
    try:
        # Write binary mode
        with open('binary.bin', 'wb') as file:
            file.write(b'Binary data\n')
        
        # Read+ mode (read and write)
        with open('example.txt', 'r+') as file:
            current = file.read()
            file.seek(0)  # Go to start
            file.write('New content\n' + current)
        
        # Exclusive creation
        with open('new_file.txt', 'x') as file:
            file.write('Created new file!\n')
            
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except PermissionError as e:
        print(f"Permission denied: {e}")
    except FileExistsError as e:
        print(f"File already exists: {e}")
    finally:
        print("File operations completed.")

# Common file modes:
# 'r': read (default)
# 'w': write (truncate)
# 'a': append
# 'x': exclusive creation
# 'b': binary mode
# '+': read and write
```

## 2. JSON File Handling

### Working with JSON Data
```python
import json
from typing import Dict, List, Any
from datetime import datetime

class DataManager:
    def __init__(self, filename: str):
        self.filename = filename
        self.data: Dict[str, Any] = {}
    
    def save_data(self, data: Dict[str, Any]) -> None:
        """Save data to JSON file."""
        try:
            # Add timestamp
            data['timestamp'] = datetime.now().isoformat()
            
            with open(self.filename, 'w') as file:
                json.dump(data, file, indent=4)
            print(f"Data saved to {self.filename}")
            
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def load_data(self) -> Dict[str, Any]:
        """Load data from JSON file."""
        try:
            with open(self.filename, 'r') as file:
                self.data = json.load(file)
            return self.data
            
        except FileNotFoundError:
            print(f"File {self.filename} not found")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return {}
    
    def update_data(self, key: str, value: Any) -> None:
        """Update specific data field."""
        try:
            data = self.load_data()
            data[key] = value
            self.save_data(data)
            
        except Exception as e:
            print(f"Error updating data: {e}")

# Demonstrate JSON handling
def demonstrate_json_handling():
    # Create data manager
    manager = DataManager('config.json')
    
    # Initial data
    user_data = {
        'name': 'John Doe',
        'age': 30,
        'preferences': {
            'theme': 'dark',
            'notifications': True
        },
        'recent_files': [
            'doc1.txt',
            'doc2.txt'
        ]
    }
    
    # Save data
    manager.save_data(user_data)
    
    # Load and print data
    loaded_data = manager.load_data()
    print("\nLoaded data:")
    print(json.dumps(loaded_data, indent=2))
    
    # Update specific field
    manager.update_data('age', 31)
    
    # Load and print updated data
    updated_data = manager.load_data()
    print("\nUpdated data:")
    print(json.dumps(updated_data, indent=2))

# Run the demonstration
demonstrate_json_handling()
```

## 3. CSV File Handling

### Working with CSV Data
```python
import csv
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SalesRecord:
    date: datetime
    product: str
    quantity: int
    price: float
    
    @property
    def total(self) -> float:
        return self.quantity * self.price

class SalesManager:
    def __init__(self, filename: str):
        self.filename = filename
    
    def save_records(self, records: List[SalesRecord]) -> None:
        """Save sales records to CSV file."""
        try:
            with open(self.filename, 'w', newline='') as file:
                writer = csv.writer(file)
                # Write header
                writer.writerow(['Date', 'Product', 'Quantity', 'Price', 'Total'])
                
                # Write records
                for record in records:
                    writer.writerow([
                        record.date.strftime('%Y-%m-%d'),
                        record.product,
                        record.quantity,
                        f"${record.price:.2f}",
                        f"${record.total:.2f}"
                    ])
            print(f"Sales records saved to {self.filename}")
            
        except Exception as e:
            print(f"Error saving records: {e}")
    
    def load_records(self) -> List[SalesRecord]:
        """Load sales records from CSV file."""
        records = []
        try:
            with open(self.filename, 'r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    record = SalesRecord(
                        date=datetime.strptime(row['Date'], '%Y-%m-%d'),
                        product=row['Product'],
                        quantity=int(row['Quantity']),
                        price=float(row['Price'].replace('$', ''))
                    )
                    records.append(record)
            return records
            
        except FileNotFoundError:
            print(f"File {self.filename} not found")
            return []
        except Exception as e:
            print(f"Error loading records: {e}")
            return []
    
    def add_record(self, record: SalesRecord) -> None:
        """Add a new sales record."""
        records = self.load_records()
        records.append(record)
        self.save_records(records)

# Demonstrate CSV handling
def demonstrate_csv_handling():
    # Create sales manager
    manager = SalesManager('sales.csv')
    
    # Create sample records
    records = [
        SalesRecord(
            date=datetime.now(),
            product="Laptop",
            quantity=2,
            price=999.99
        ),
        SalesRecord(
            date=datetime.now(),
            product="Mouse",
            quantity=5,
            price=29.99
        ),
        SalesRecord(
            date=datetime.now(),
            product="Keyboard",
            quantity=3,
            price=59.99
        )
    ]
    
    # Save records
    manager.save_records(records)
    
    # Load and print records
    loaded_records = manager.load_records()
    print("\nLoaded sales records:")
    for record in loaded_records:
        print(f"{record.product}: {record.quantity} units at ${record.price:.2f} each")
        print(f"Total: ${record.total:.2f}")
    
    # Add new record
    new_record = SalesRecord(
        date=datetime.now(),
        product="Monitor",
        quantity=1,
        price=299.99
    )
    manager.add_record(new_record)

# Run the demonstration
demonstrate_csv_handling()
```

## 4. Best Practices and Tips

### File Handling Best Practices
1. **Always Use Context Managers**
   - Use `with` statement for automatic file closing
   - Ensures proper resource cleanup

2. **Error Handling**
   - Handle specific exceptions
   - Provide meaningful error messages
   - Clean up resources in `finally` block

3. **Path Handling**
   - Use `pathlib` for cross-platform compatibility
   - Always use raw strings for Windows paths
   - Validate paths before operations

4. **Performance Considerations**
   - Use appropriate buffer sizes
   - Consider memory usage for large files
   - Use generators for line-by-line processing

### Example: Safe File Operations
```python
from pathlib import Path
from typing import Generator, Optional
import shutil

class FileHandler:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def safe_write(self, filename: str, content: str) -> bool:
        """Safely write content to file with backup."""
        file_path = self.base_dir / filename
        backup_path = self.base_dir / f"{filename}.bak"
        
        try:
            # Create backup if file exists
            if file_path.exists():
                shutil.copy2(file_path, backup_path)
            
            # Write new content
            file_path.write_text(content)
            return True
            
        except Exception as e:
            print(f"Error writing file: {e}")
            # Restore from backup if available
            if backup_path.exists():
                shutil.copy2(backup_path, file_path)
            return False
        finally:
            # Clean up backup
            if backup_path.exists():
                backup_path.unlink()
    
    def read_chunks(self, filename: str, chunk_size: int = 1024) -> Generator[str, None, None]:
        """Read file in chunks to handle large files."""
        file_path = self.base_dir / filename
        try:
            with open(file_path, 'r') as file:
                while True:
                    chunk = file.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
                    
        except Exception as e:
            print(f"Error reading file: {e}")
            yield ""
    
    def find_files(self, pattern: str) -> Generator[Path, None, None]:
        """Find files matching pattern."""
        try:
            yield from self.base_dir.glob(pattern)
        except Exception as e:
            print(f"Error finding files: {e}")

# Demonstrate safe file operations
def demonstrate_safe_operations():
    handler = FileHandler('data')
    
    # Write content safely
    content = "Important data\nMore data\nEven more data"
    success = handler.safe_write('important.txt', content)
    print(f"Write successful: {success}")
    
    # Read in chunks
    print("\nReading in chunks:")
    for chunk in handler.read_chunks('important.txt'):
        print(f"Chunk: {chunk.strip()}")
    
    # Find files
    print("\nFinding text files:")
    for file_path in handler.find_files('*.txt'):
        print(f"Found: {file_path}")

# Run the demonstration
demonstrate_safe_operations()
```

## Summary

### Key Points
1. Always use context managers (`with` statement)
2. Handle file operations errors appropriately
3. Use appropriate file modes and encodings
4. Consider performance for large files
5. Keep backups for important operations

### What's Next
- [Testing and Debugging](06-Python-Testing-Debugging.md)
- [Web Development](07-Python-Web-Development.md)

---

> **Navigation**
> - [← Testing and Debugging](06-Python-Testing-Debugging.md)
> - [Web Development →](07-Python-Web-Development.md)
