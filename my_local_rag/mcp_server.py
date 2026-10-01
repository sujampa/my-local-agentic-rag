from pathlib import Path

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("My Local RAG Tools")


@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    """
    Add two numbers together.
    """
    return a + b


@mcp.tool()
def get_document_list() -> list[str]:
    """
    Return the names of all PDF documents available
    in the local documents folder.
    """

    documents_folder = Path("documents")

    if not documents_folder.exists():
        return []

    pdf_files = sorted(
        file.name
        for file in documents_folder.glob("*.pdf")
    )

    return pdf_files


if __name__ == "__main__":
    mcp.run()