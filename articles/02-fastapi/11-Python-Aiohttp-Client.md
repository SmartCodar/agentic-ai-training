# Day - 11 Aiohttp Client Deep Dive

## Overview
This lesson explores aiohttp, a powerful asynchronous HTTP client/server framework for Python. You'll learn how to make efficient HTTP requests, handle responses, and build robust async applications.

## Learning Objectives
- Master aiohttp client functionality
- Implement concurrent HTTP requests
- Handle errors and timeouts
- Create real-world async clients
- Understand session management
- Build practical web scrapers and API clients

## Prerequisites
- Understanding of async/await in Python
- Knowledge of HTTP protocols and requests
- Familiarity with JSON data handling
- Python 3.7+ installed

## Time Estimate
- Reading: 40 minutes
- Practice: 60 minutes
- Projects: 50 minutes

---

## 1. Getting Started with Aiohttp

### Installation and Basic Setup
```bash
# Install aiohttp
pip install aiohttp[speedups]

# For development, also install these
pip install aiohttp-devtools
pip install pytest-aiohttp
```

### Basic HTTP Methods
```python
import aiohttp
import asyncio
from typing import Dict, Any, Optional
import json

async def basic_requests():
    async with aiohttp.ClientSession() as session:
        # GET request
        async with session.get('https://api.github.com/events') as response:
            print(f"GET Status: {response.status}")
            data = await response.json()
        
        # POST request
        async with session.post('https://httpbin.org/post',
                              json={'key': 'value'}) as response:
            print(f"POST Status: {response.status}")
            data = await response.json()
        
        # PUT request
        async with session.put('https://httpbin.org/put',
                             data='update') as response:
            print(f"PUT Status: {response.status}")
        
        # DELETE request
        async with session.delete('https://httpbin.org/delete') as response:
            print(f"DELETE Status: {response.status}")

# Run example
asyncio.run(basic_requests())
```

## 2. Advanced HTTP Client

### Robust API Client
```python
from dataclasses import dataclass
from datetime import datetime
import asyncio
import aiohttp
from typing import List, Dict, Optional, Any
import logging

@dataclass
class APIResponse:
    status: int
    data: Any
    headers: Dict
    timestamp: datetime

class AsyncAPIClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
    
    async def __aenter__(self):
        """Setup client session."""
        self.session = aiohttp.ClientSession(
            headers=self._get_headers()
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup client session."""
        if self.session:
            await self.session.close()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers
    
    async def _request(self, method: str, endpoint: str,
                      **kwargs) -> APIResponse:
        """Make HTTP request with error handling."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with'")
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                data = await response.json()
                return APIResponse(
                    status=response.status,
                    data=data,
                    headers=dict(response.headers),
                    timestamp=datetime.now()
                )
        except aiohttp.ClientError as e:
            self.logger.error(f"Request failed: {e}")
            raise
    
    async def get(self, endpoint: str, params: Optional[Dict] = None) -> APIResponse:
        """Send GET request."""
        return await self._request('GET', endpoint, params=params)
    
    async def post(self, endpoint: str, data: Dict) -> APIResponse:
        """Send POST request."""
        return await self._request('POST', endpoint, json=data)
    
    async def put(self, endpoint: str, data: Dict) -> APIResponse:
        """Send PUT request."""
        return await self._request('PUT', endpoint, json=data)
    
    async def delete(self, endpoint: str) -> APIResponse:
        """Send DELETE request."""
        return await self._request('DELETE', endpoint)

# Example usage
async def main():
    async with AsyncAPIClient('https://api.example.com', 'your-api-key') as client:
        # GET request
        response = await client.get('/users', params={'page': 1})
        print(f"Users: {response.data}")
        
        # POST request
        new_user = {'name': 'John Doe', 'email': 'john@example.com'}
        response = await client.post('/users', data=new_user)
        print(f"Created user: {response.data}")

# Run example
asyncio.run(main())
```

## 3. Concurrent Web Scraper

### Advanced Web Scraper
```python
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import List, Set, Dict, Optional
import time
import logging
from urllib.parse import urljoin

@dataclass
class ScrapedPage:
    url: str
    title: str
    links: Set[str]
    status: int
    timestamp: datetime

class AsyncWebScraper:
    def __init__(self, base_url: str, max_pages: int = 10,
                 max_workers: int = 3):
        self.base_url = base_url
        self.max_pages = max_pages
        self.max_workers = max_workers
        self.visited: Set[str] = set()
        self.to_visit: Set[str] = {base_url}
        self.results: List[ScrapedPage] = []
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
    
    async def __aenter__(self):
        """Initialize session."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup session."""
        if self.session:
            await self.session.close()
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL should be scraped."""
        return (
            url.startswith(self.base_url) and
            url not in self.visited and
            len(self.visited) < self.max_pages
        )
    
    async def _fetch_page(self, url: str) -> Optional[ScrapedPage]:
        """Fetch and parse a single page."""
        if not self.session:
            raise RuntimeError("Scraper not initialized. Use 'async with'")
        
        try:
            async with self.session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract title
                title = soup.title.string if soup.title else "No title"
                
                # Extract links
                links = set()
                for link in soup.find_all('a', href=True):
                    full_url = urljoin(url, link['href'])
                    if self._is_valid_url(full_url):
                        links.add(full_url)
                        self.to_visit.add(full_url)
                
                return ScrapedPage(
                    url=url,
                    title=title,
                    links=links,
                    status=response.status,
                    timestamp=datetime.now()
                )
        
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {e}")
            return None
    
    async def _worker(self, worker_id: int):
        """Worker to process URLs from the queue."""
        while self.to_visit and len(self.visited) < self.max_pages:
            try:
                url = self.to_visit.pop()
                if url not in self.visited:
                    self.logger.info(f"Worker {worker_id} processing: {url}")
                    if result := await self._fetch_page(url):
                        self.results.append(result)
                    self.visited.add(url)
            except KeyError:  # to_visit was empty
                break
    
    async def scrape(self) -> List[ScrapedPage]:
        """Start the scraping process."""
        workers = [
            self._worker(i) for i in range(self.max_workers)
        ]
        await asyncio.gather(*workers)
        return self.results

# Example usage
async def main():
    start_time = time.time()
    
    async with AsyncWebScraper('https://example.com', max_pages=5) as scraper:
        results = await scraper.scrape()
        
        print(f"\nScraped {len(results)} pages in "
              f"{time.time() - start_time:.2f} seconds:")
        
        for page in results:
            print(f"\nURL: {page.url}")
            print(f"Title: {page.title}")
            print(f"Found {len(page.links)} links")

# Run scraper
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
```

## 4. Rate-Limited API Client

### Rate-Limited Client Implementation
```python
import aiohttp
import asyncio
from typing import Dict, Optional, Any
import time
from dataclasses import dataclass
import logging

@dataclass
class RateLimitConfig:
    calls: int
    period: float

class RateLimitedClient:
    def __init__(self, base_url: str, rate_limit: RateLimitConfig):
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.calls_made = 0
        self.period_start = time.time()
        self.lock = asyncio.Lock()
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
    
    async def __aenter__(self):
        """Initialize session."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup session."""
        if self.session:
            await self.session.close()
    
    async def _wait_for_rate_limit(self):
        """Implement rate limiting."""
        async with self.lock:
            current_time = time.time()
            time_passed = current_time - self.period_start
            
            if time_passed > self.rate_limit.period:
                # Reset period
                self.period_start = current_time
                self.calls_made = 0
            elif self.calls_made >= self.rate_limit.calls:
                # Wait for next period
                sleep_time = self.rate_limit.period - time_passed
                self.logger.info(f"Rate limit reached. "
                               f"Waiting {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
                self.period_start = time.time()
                self.calls_made = 0
            
            self.calls_made += 1
    
    async def request(self, method: str, endpoint: str,
                     **kwargs) -> Dict[str, Any]:
        """Make rate-limited request."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with'")
        
        await self._wait_for_rate_limit()
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        async with self.session.request(method, url, **kwargs) as response:
            return await response.json()

# Example usage
async def main():
    # Configure client (5 calls per 2 seconds)
    config = RateLimitConfig(calls=5, period=2.0)
    
    async with RateLimitedClient('https://api.example.com', config) as client:
        # Make multiple requests
        tasks = []
        for i in range(10):
            task = client.request(
                'GET',
                f'/items/{i}',
                params={'detail': 'full'}
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        print(f"Completed {len(results)} requests")

# Run example
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
```

## 5. Best Practices

### Session Management
1. Always use context managers
2. One session per application
3. Close sessions properly

### Error Handling
1. Handle network errors
2. Implement retries
3. Use timeouts
4. Log failures

### Performance
1. Use connection pooling
2. Implement rate limiting
3. Handle concurrent requests
4. Monitor memory usage

## Summary

### Key Takeaways
1. Aiohttp provides powerful async HTTP capabilities
2. Session management is crucial
3. Error handling and rate limiting are essential
4. Concurrent requests need careful management

### What's Next
- [FastAPI Development](12-Python-FastAPI.md)
- WebSocket Programming
- Advanced HTTP Protocols

---

> **Navigation**
> - [← Async Programming](10-Python-Async-Programming.md)
> - [FastAPI →](12-Python-FastAPI.md)
