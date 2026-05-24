# MathOps

A simple stateless calculator.

## Operations

- `add(a, b)` — sum of two numbers
- `divide(a, b)` — `a / b`, raises `ZeroDivisionError` when `b == 0`
- `multiply(a, b)` — product of two numbers
- `sqrt(x)` — square root of a non-negative number
- `subtract(a, b)` — `a - b`

## Example

```python
from calculator import MathOps
c = MathOps()
c.add(2, 3)       # 5
c.multiply(4, 5)  # 20
```
