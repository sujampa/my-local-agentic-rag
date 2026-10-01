import asyncio
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():

    # Get the folder where this client file is located
    project_folder = Path(__file__).parent

    # Get the exact MCP server path
    server_file = project_folder / "mcp_server.py"

    server_params = StdioServerParameters(
        command="python",
        args=[str(server_file)],
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            # Initialize MCP connection
            await session.initialize()

            print("\nConnected to MCP server!")

            # Discover available tools
            tools = await session.list_tools()

            print("\nAvailable MCP tools:")

            for tool in tools.tools:

                print(
                    f"- {tool.name}: "
                    f"{tool.description}"
                )

            # Test document-list tool
            print("\nCalling get_document_list...")

            result = await session.call_tool(
                "get_document_list",
                arguments={}
            )

            print("\nDocuments:")

            print(result)


if __name__ == "__main__":
    asyncio.run(main())