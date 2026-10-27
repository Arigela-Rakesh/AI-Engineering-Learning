from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Employee Handbook Tools")


@mcp.tool()
def calculate(
    a: float,
    b: float,
    operation: str,
) -> float:
    """Perform basic arithmetic on two numbers."""

    if operation == "add":
        return a + b

    if operation == "subtract":
        return a - b

    if operation == "multiply":
        return a * b

    if operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b

    raise ValueError(
        "Unsupported operation. Use add, subtract, multiply, or divide."
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")