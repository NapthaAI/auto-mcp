import inspect
import asyncio
import functools
import sys
import contextlib
import io
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union

from mcp.server.fastmcp import FastMCP
from mcp.types import Tool as MCPTool
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, create_model

# Suppress warnings globally for the module
warnings.filterwarnings("ignore")

class BaseMCPAdapter:
    """Base adapter for converting agent frameworks to MCP servers."""
    
    def __init__(
        self,
        name: str,
        input_schema: Type[BaseModel],
        transport: Literal["stdio", "sse"] = "stdio",
        host: str = "localhost",
        port: int = 8000,
        description: str = None,
        debug: bool = False,
    ):
        """Initialize the base adapter with common parameters."""
        self.name = name
        self.input_schema = input_schema
        self.description = description or f"MCP server for {name}"
        self.transport = transport
        self.host = host
        self.port = port
        self.debug = debug
        self.mcp_server = FastMCP(name)
        self._setup_mcp_server()
    
    def _log(self, message: str):
        """Log a message if debug is enabled."""
        if self.debug:
            print(f"DEBUG: {message}", file=sys.__stderr__)  # Write to real stderr
    
    def _setup_mcp_server(self):
        """Set up the MCP server with tools."""
        @self.mcp_server.tool()
        def list_tools() -> List[MCPTool]:
            return self._get_tools()
        
        @self.mcp_server.tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            return await self._handle_tool_call(name, arguments)
    
    def _get_tools(self) -> List[MCPTool]:
        """Get the list of tools for this adapter."""
        schema_fields = {}
        for field_name, field_info in self.input_schema.model_fields.items():
            required = field_info.is_required()
            schema_fields[field_name] = {
                "type": self._get_json_type(field_info.annotation),
                "description": getattr(field_info, "description", ""),
                "required": required
            }
        
        tool = MCPTool(
            name=self.name,
            description=self.description,
            inputSchema=schema_fields
        )
        
        return [tool]
    
    def _get_json_type(self, python_type):
        """Convert Python type to JSON schema type."""
        type_mapping = {
            "int": "integer",
            "float": "number",
            "str": "string",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
        }
        
        if hasattr(python_type, "__name__"):
            type_name = python_type.__name__.lower()
        else:
            type_name = str(python_type).lower()
        
        return type_mapping.get(type_name, "string")
    
    async def _handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle a tool call."""
        raise NotImplementedError("Subclasses must implement _handle_tool_call")
    
    def run(self):
        """Run the MCP server."""
        # Save the original stdout and stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        try:
            print(f"\n🚀 Starting MCP server '{self.name}' with {self.transport} transport", 
                  file=sys.__stdout__)  # Write to real stdout
            
            if self.transport == "stdio":
                # When using stdio transport, we need to be extra careful
                # Run the server in a context that captures unexpected output
                self.mcp_server.run(transport="stdio")
            elif self.transport == "sse":
                print(f"📡 Listening on http://{self.host}:{self.port}", 
                      file=sys.__stdout__)  # Write to real stdout
                self.mcp_server.run(transport="sse", host=self.host, port=self.port)
            else:
                raise ValueError(f"Unsupported transport: {self.transport}")
        except Exception as e:
            import traceback
            print(f"ERROR in run: {str(e)}\n{traceback.format_exc()}", 
                  file=sys.__stderr__)  # Write to real stderr
        finally:
            # Restore the original stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr

# Import specific adapters
from .adapters.crewai_adapter import CrewAIAdapter, crewai_mcp

# Define what should be available when importing the package
__all__ = ["BaseMCPAdapter", "CrewAIAdapter", "crewai_mcp"]