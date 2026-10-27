import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server_path = Path(__file__).with_name("server.py")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()

            print("Available tools:")

            for tool in tools_result.tools:
                print(f"- {tool.name}: {tool.description}")

            result = await session.call_tool(
                "calculate",
                {
                    "a": 25,
                    "b": 15,
                    "operation": "add",
                },
            )

            print("\nCalculation result:")

            for item in result.content:
                if hasattr(item, "text"):
                    print(item.text)

            if result.isError:
                raise RuntimeError("MCP tool execution failed.")


if __name__ == "__main__":
    asyncio.run(main())