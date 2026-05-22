"""Demo: structured output — extract structured data from free text.

The agent reads a product description and returns a validated Pydantic model.
No tools needed; the runtime injects the JSON schema as a system message and
validates the final response.

Run: uv run python demos/demo_structured_output.py
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from harness.adapters.ollama import OllamaAdapter
from harness.core import Agent, Done, FailoverPolicy, RunRequest, ToolRegistry
from harness.storage.memory import InMemoryStorage


class ProductInfo(BaseModel):
    name: str
    price_usd: float
    category: str
    in_stock: bool
    rating: float  # 0.0 to 5.0


MODEL = "gemma4:latest"

DESCRIPTION = """
The AeroGlide Pro X7 is a high-performance carbon-fibre racing bicycle priced at $3,249.
It sits in the road cycling category and currently has strong inventory at most retailers.
Customers give it an average of 4.6 out of 5 stars for its lightweight frame and
precision gearing system.
"""


async def main() -> None:
    adapter = OllamaAdapter()
    storage = InMemoryStorage()
    registry = ToolRegistry()
    failover = FailoverPolicy(chain=["ollama"], max_attempts=1)

    agent = Agent(
        adapters={"ollama": adapter},
        tools=registry,
        storage=storage,
        failover=failover,
        default_model=MODEL,
    )

    print(f"Model: {MODEL}")
    print(f"Input: {DESCRIPTION.strip()}\n")

    product: ProductInfo | None = None
    async for event in agent.run(
        RunRequest(
            prompt=f"Extract product information from this description:\n{DESCRIPTION}",
            result_type=ProductInfo,
        )
    ):
        if isinstance(event, Done) and event.structured_result:
            product = ProductInfo(**event.structured_result)

    if product:
        print("Extracted ProductInfo:")
        print(f"  name       : {product.name}")
        print(f"  price_usd  : ${product.price_usd:,.2f}")
        print(f"  category   : {product.category}")
        print(f"  in_stock   : {product.in_stock}")
        print(f"  rating     : {product.rating}/5.0")
    else:
        print("No structured result produced.")


if __name__ == "__main__":
    asyncio.run(main())
