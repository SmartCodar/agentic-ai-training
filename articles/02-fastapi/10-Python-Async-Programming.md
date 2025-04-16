# Day - 10 Introduction to Asynchronous Programming

## Overview
This lesson explores asynchronous programming in Python using async/await and event loops. You'll learn how to write non-blocking code that can handle multiple operations efficiently, making your applications more scalable and responsive.

## Learning Objectives
- Understand synchronous vs asynchronous execution
- Master async/await syntax
- Work with event loops
- Implement concurrent operations
- Handle async context managers
- Create real-world async applications

## Prerequisites
- Strong understanding of Python fundamentals
- Knowledge of functions and error handling
- Understanding of basic threading concepts
- Python 3.7+ installed
- Completion of [07 - Testing and Debugging](../01-basics/07-Python-Testing-Debugging.md)
- Completion of [08 - Functional Programming](../01-basics/08-Python-Functional-Programming.md)
- Completion of [09 - Project Setup](../01-basics/09-Python-Project-Setup.md)

## Time Estimate
- Reading: 45 minutes
- Practice: 60 minutes
- Exercises: 45 minutes

## ⚙️ Setup Check
```python
import sys
import asyncio

async def check_async_support():
    try:
        # Check Python version
        print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
        if sys.version_info < (3, 7):
            print("Warning: Python 3.7+ recommended for full async support")
        
        # Verify asyncio
        await asyncio.sleep(0.1)
        print("Asyncio working correctly")
        
        # Check event loop
        loop = asyncio.get_running_loop()
        print("Event loop available")
        
        return True
    except Exception as e:
        print(f"Setup check failed: {e}")
        return False

# Run setup check
if __name__ == "__main__":
    asyncio.run(check_async_support())
```

---

## 1. Understanding Asynchronous Programming

### Synchronous vs Asynchronous
```python
# Synchronous example
def sync_task():
    print("Start")
    time.sleep(2)  # Blocks execution
    print("End")

# Asynchronous example
async def async_task():
    print("Start")
    await asyncio.sleep(2)  # Non-blocking
    print("End")
```

### Real-world Example: Weather API Client
```python
import asyncio
import aiohttp
from dataclasses import dataclass
from typing import List, Dict, Optional
import json
import time

@dataclass
class WeatherData:
    city: str
    temperature: float
    conditions: str
    timestamp: str

class AsyncWeatherClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.weatherapi.com/v1"
    
    async def get_weather(self, city: str) -> Optional[WeatherData]:
        """Get weather data for a city asynchronously."""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/current.json"
            params = {
                "key": self.api_key,
                "q": city
            }
            
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return WeatherData(
                            city=city,
                            temperature=data["current"]["temp_c"],
                            conditions=data["current"]["condition"]["text"],
                            timestamp=data["current"]["last_updated"]
                        )
                    else:
                        print(f"Error fetching data for {city}: {response.status}")
                        return None
            except Exception as e:
                print(f"Exception while fetching data for {city}: {e}")
                return None
    
    async def get_multiple_weather(self, cities: List[str]) -> List[WeatherData]:
        """Get weather data for multiple cities concurrently."""
        tasks = [self.get_weather(city) for city in cities]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

# Example usage
async def main():
    # Initialize client
    client = AsyncWeatherClient("your-api-key")
    
    # Cities to check
    cities = ["London", "New York", "Tokyo", "Sydney", "Paris"]
    
    # Measure time for async operation
    start_time = time.time()
    weather_data = await client.get_multiple_weather(cities)
    end_time = time.time()
    
    # Print results
    print(f"\nWeather data fetched in {end_time - start_time:.2f} seconds:")
    for data in weather_data:
        print(f"\n{data.city}:")
        print(f"  Temperature: {data.temperature}°C")
        print(f"  Conditions: {data.conditions}")
        print(f"  Updated: {data.timestamp}")

# Run the async program
if __name__ == "__main__":
    asyncio.run(main())
```

## 2. Event Loops and Task Management

### Understanding Event Loops
The event loop is the core of every async application. It manages and distributes tasks efficiently.

```python
import asyncio
from typing import List, Any
import random

class TaskManager:
    def __init__(self):
        self.tasks: List[asyncio.Task] = []
    
    async def create_task(self, name: str, duration: float) -> Any:
        """Simulate an async task."""
        print(f"Starting task: {name}")
        await asyncio.sleep(duration)
        result = random.randint(1, 100)
        print(f"Completed task: {name} with result: {result}")
        return result
    
    def add_task(self, name: str, duration: float) -> None:
        """Add a task to the event loop."""
        task = asyncio.create_task(self.create_task(name, duration))
        self.tasks.append(task)
    
    async def run_all(self) -> List[Any]:
        """Run all tasks concurrently."""
        results = await asyncio.gather(*self.tasks)
        return results

async def demonstrate_event_loop():
    # Create task manager
    manager = TaskManager()
    
    # Add various tasks
    manager.add_task("Database Query", 2.0)
    manager.add_task("API Call", 1.5)
    manager.add_task("File Processing", 1.0)
    manager.add_task("Data Analysis", 2.5)
    
    # Run all tasks
    print("\nStarting all tasks...")
    start_time = time.time()
    
    results = await manager.run_all()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nAll tasks completed in {total_time:.2f} seconds")
    print(f"Results: {results}")

# Run demonstration
if __name__ == "__main__":
    asyncio.run(demonstrate_event_loop())
```

## 3. Async Context Managers

### Understanding Async Context Managers
Async context managers help manage resources in async code, similar to regular context managers but for async operations.

```python
import asyncio
from typing import AsyncGenerator
from contextlib import asynccontextmanager

class AsyncResource:
    async def initialize(self):
        print("Initializing resource...")
        await asyncio.sleep(1)
        print("Resource initialized")
    
    async def cleanup(self):
        print("Cleaning up resource...")
        await asyncio.sleep(0.5)
        print("Resource cleaned up")

@asynccontextmanager
async def managed_resource() -> AsyncGenerator[AsyncResource, None]:
    """Async context manager for resource management."""
    resource = AsyncResource()
    try:
        await resource.initialize()
        yield resource
    finally:
        await resource.cleanup()

async def process_data():
    async with managed_resource() as resource:
        print("Processing with resource...")
        await asyncio.sleep(2)
        print("Processing complete")

# Run example
if __name__ == "__main__":
    asyncio.run(process_data())
```

## 4. Best Practices

### When to Use Async
1. I/O-bound operations:
   - Network requests
   - File operations
   - Database queries

2. Multiple independent operations:
   - Parallel API calls
   - Batch processing
   - Concurrent file operations

### When Not to Use Async
1. CPU-bound tasks:
   - Complex calculations
   - Image processing
   - Data analysis

2. Simple sequential operations:
   - Basic file reading
   - Single API call
   - Linear data processing

### Error Handling
```python
async def safe_operation():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://api.example.com') as response:
                return await response.json()
    except aiohttp.ClientError as e:
        print(f"Network error: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None
```

## Summary

## 5. Practical Exercises 🔨

### Exercise 1: Async Web Scraper
Create a simple web scraper that fetches titles from multiple web pages concurrently:
```python
async def fetch_page_title(url: str) -> str:
    """Fetch and extract title from a webpage."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            html = await response.text()
            # Extract title using string operations
            title_start = html.find('<title>') + 7
            title_end = html.find('</title>')
            return html[title_start:title_end]
```

### Exercise 2: File Processor
Implement an async file processor that reads multiple files concurrently:
```python
async def process_file(path: str) -> dict:
    """Process a file asynchronously."""
    async with aiofiles.open(path, 'r') as file:
        content = await file.read()
        return {
            'path': path,
            'size': len(content),
            'lines': len(content.splitlines())
        }
```

### Exercise 3: Database Client
Create an async database client using aiosqlite:
```python
async def query_db(query: str) -> List[dict]:
    """Execute a database query asynchronously."""
    async with aiosqlite.connect('example.db') as db:
        async with db.execute(query) as cursor:
            rows = await cursor.fetchall()
            return rows
```

## 6. Knowledge Check ✅

1. What is asynchronous programming and how does it differ from synchronous code?
2. Explain the role of the event loop in async programming.
3. What is the purpose of `await` and when should you use it?
4. How do you run multiple async tasks concurrently?
5. What are async context managers and when should you use them?
6. How do you handle errors in async code?
7. What types of operations are best suited for async programming?
8. How do you convert a synchronous program to use async/await?

## 7. Summary

### Key Takeaways
1. Async programming improves application efficiency
2. Event loops manage concurrent operations
3. Async context managers handle resource lifecycle
4. Best practices guide when to use async

## 📚 Additional Resources
- [Python asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [Real Python - Async IO Guide](https://realpython.com/async-io-python/)
- [AsyncIO for the Working Python Developer](https://www.youtube.com/watch?v=iG6fr81xHKA)
- [Python Async Design Patterns](https://docs.python.org/3/library/asyncio-dev.html)

---

> **Navigation**
> - [← Project Setup](../01-basics/09-Python-Project-Setup.md)
> - [Aiohttp Client →](11-Python-Aiohttp-Client.md)
