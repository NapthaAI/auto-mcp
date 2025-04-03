import warnings
from typing import Any
from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP
from auto_mcp.adapters.crewai_adapter import crewai_to_mcp_tool
from stock_analysis.crew import StockAnalysisCrew


class QueryInput(BaseModel):
    query: str = Field(..., description="The query to ask the financial agent")
    company_stock: str = Field(default="AMZN", description="The company stock ticker symbol")
    # Using Pydantic V2 style config
    model_config = ConfigDict(extra="forbid")


mcp = FastMCP("Stock Analysis MCP Server")
name = "financial_agent"
description = "A financial agent that analyzes stocks and provides investment recommendations"
input_schema = QueryInput

# Create the tool using the function-based approach
tool = crewai_to_mcp_tool(
    crewai_class=StockAnalysisCrew,
    name=name,
    description=description,
    input_schema=input_schema,
)

# Add the tool to the MCP server
mcp.add_tool(
    tool,
    name=name,
    description=description,
)


def serve_sse():
    """Start the MCP server using SSE transport"""
    mcp.run(transport="sse")


def serve_stdio():
    """Start the MCP server using stdio transport"""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    serve_stdio()