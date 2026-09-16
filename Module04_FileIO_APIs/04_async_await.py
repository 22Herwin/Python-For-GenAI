# Module 04 - File I/O & APIs
# 4.2 Async/Await
# When you need to call multiple APIs concurrently - comparing model outputs,
# embedding a batch of documents, or fetching data while the LLM responds - use
# asyncio. The httpx library is the modern async-capable HTTP client.
#
# NOTE: this script makes real network calls to a public test API
# (jsonplaceholder.typicode.com). It requires internet access to run.

import asyncio
import httpx


async def fetch_json(client: httpx.AsyncClient, url: str) -> dict:
    """Fetch a URL asynchronously and return JSON."""
    response = await client.get(url, timeout=10.0)
    response.raise_for_status()
    return response.json()


async def fetch_multiple(urls: list[str]) -> list[dict]:
    """Fetch all URLs concurrently - much faster than sequential."""
    async with httpx.AsyncClient() as client:
        tasks = [fetch_json(client, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]


async def main():
    # JSONPlaceholder - a free public test API
    urls = [
        f"https://jsonplaceholder.typicode.com/posts/{i}"
        for i in range(1, 4)
    ]
    posts = await fetch_multiple(urls)
    for post in posts:
        print(f"Post {post['id']}: {post['title'][:40]}")


if __name__ == "__main__":
    asyncio.run(main())
