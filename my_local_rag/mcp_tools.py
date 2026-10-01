import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ============================================================
# RUN AN MCP TOOL
# ============================================================

async def call_mcp_tool(tool_name, arguments):

    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(
            read,
            write
        ) as session:

            await session.initialize()

            result = await session.call_tool(
                tool_name,
                arguments=arguments
            )

            return result


def run_mcp_tool(tool_name, arguments):

    return asyncio.run(
        call_mcp_tool(
            tool_name,
            arguments
        )
    )


# ============================================================
# MCP: ADD NUMBERS
# ============================================================

def mcp_add_numbers(a, b):

    result = run_mcp_tool(
        "add_numbers",
        {
            "a": a,
            "b": b
        }
    )

    if result.structuredContent:

        return result.structuredContent

    if result.content:

        return result.content[0].text

    return result


# ============================================================
# MCP: GET DOCUMENT LIST
# ============================================================

def mcp_get_document_list():

    result = run_mcp_tool(
        "get_document_list",
        {}
    )

    if result.structuredContent:

        return result.structuredContent

    if result.content:

        return result.content[0].text

    return result