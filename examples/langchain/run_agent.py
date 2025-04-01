from langchain_agent_server import ask_pplx, logger, QueryInput
from auto_mcp import langchain_mcp_from_func

def main():
    """
    Create and run an MCP server using the function-based approach.
    """
    logger.info("Creating MCP server with function-based approach")
    mcp_server = langchain_mcp_from_func(
        func=ask_pplx,
        input_schema=QueryInput,
        name="Perplexity Agent",
        description="Process queries using the Perplexity language model",
        debug=True
    )
    
    logger.info("Starting MCP server with stdio transport")
    mcp_server.run(transport="stdio")
    logger.info("MCP server stopped")

if __name__ == "__main__":
    main()
