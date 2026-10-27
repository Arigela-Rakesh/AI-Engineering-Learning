import asyncio
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    server_path = Path(__file__).with_name("server.py")

    client = MultiServerMCPClient(
        {
            "calculator": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(server_path)],
            }
        }
    )

    tools = await client.get_tools()

    print("Discovered MCP tools:")

    for tool in tools:
        print(f"- {tool.name}: {tool.description}")

    calculate_tool = next(
        (
            tool
            for tool in tools
            if tool.name == "calculate"
        ),
        None,
    )

    if calculate_tool is None:
        raise RuntimeError(
            "The calculate MCP tool was not discovered."
        )

    result = await calculate_tool.ainvoke(
        {
            "a": 25,
            "b": 15,
            "operation": "add",
        }
    )

    print("\nMCP calculation result:", result)


if __name__ == "__main__":
    asyncio.run(main())