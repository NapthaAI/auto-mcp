import inspect
import asyncio
import functools
import sys
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union

from mcp.server.fastmcp import FastMCP
from mcp.types import Tool as MCPTool
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, create_model


def _get_json_type(python_type):
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


def _get_tools(name: str, description: str, input_schema: Type[BaseModel]) -> List[MCPTool]:
    """Get the list of tools based on input schema."""
    schema_fields = {}
    for field_name, field_info in input_schema.model_fields.items():
        required = field_info.is_required()
        schema_fields[field_name] = {
            "type": _get_json_type(field_info.annotation),
            "description": getattr(field_info, "description", ""),
            "required": required
        }
    
    tool = MCPTool(
        name=name,
        description=description,
        inputSchema=schema_fields
    )
    
    return [tool]


def _log(message: str, debug: bool):
    """Log a message if debug is enabled."""
    if debug:
        print(f"DEBUG: {message}")


def langchain_mcp(
    input_schema: Type[BaseModel],
    name: str,
    description: str = None,
    transport: Literal["stdio", "sse"] = "stdio",
    host: str = "localhost",
    port: int = 8000,
    debug: bool = False,
):
    """
    Decorator for converting LangChain agents to MCP servers.
    
    Args:
        input_schema: Pydantic model defining the input schema
        name: Name for the MCP server
        description: Description of the server
        transport: Transport protocol ("stdio" or "sse")
        host: Host for SSE transport
        port: Port for SSE transport
        debug: Enable debug logging
    
    Example:
        class QueryInput(BaseModel):
            query: str
        
        @langchain_mcp(input_schema=QueryInput, name="Search Agent")
        def search_agent(query: str):
            # Create and return a LangChain agent
            return create_react_agent(llm, tools)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if args or kwargs:
                return func(*args, **kwargs)
            
            # Set up description
            tool_description = description or func.__doc__ or f"MCP server for {name}"
            agent = None
            
            # Create MCP server
            mcp_server = FastMCP(name)
            
            # Define list_tools handler
            @mcp_server.tool()
            def list_tools() -> List[MCPTool]:
                return _get_tools(name, tool_description, input_schema)
            
            # Define call_tool handler
            @mcp_server.tool()
            async def call_tool(tool_name: str, arguments: Dict[str, Any]) -> CallToolResult:
                nonlocal agent
                
                _log(f"Tool call: {tool_name} with arguments: {arguments}", debug)
                
                # Handle nested call_tool pattern
                if tool_name == "call_tool" and isinstance(arguments, dict) and "name" in arguments and "arguments" in arguments:
                    nested_name = arguments["name"]
                    nested_args = arguments["arguments"]
                    _log(f"Nested call_tool - name: {nested_name}", debug)
                    return await call_tool(nested_name, nested_args)
                
                # Handle call to our main tool
                if tool_name == name:
                    try:
                        # Convert string arguments to proper input schema if needed
                        if isinstance(arguments, str):
                            arguments = {"query": arguments}
                        
                        # Validate input against schema
                        input_data = input_schema(**arguments)
                        
                        # Call the agent factory with validated input
                        if agent is None:
                            if asyncio.iscoroutinefunction(func):
                                agent = await func(**input_data.model_dump())
                            else:
                                agent = func(**input_data.model_dump())
                        
                        # Process input with the agent
                        if hasattr(agent, "invoke"):
                            if asyncio.iscoroutinefunction(agent.invoke):
                                result = await agent.invoke(input_data.model_dump())
                            else:
                                result = agent.invoke(input_data.model_dump())
                        elif hasattr(agent, "run"):
                            if asyncio.iscoroutinefunction(agent.run):
                                result = await agent.run(str(input_data.model_dump()))
                            else:
                                result = agent.run(str(input_data.model_dump()))
                        else:
                            result = str(agent)
                        
                        return CallToolResult(
                            isError=False,
                            content=[TextContent(type="text", text=str(result))]
                        )
                    except Exception as e:
                        import traceback
                        tb = traceback.format_exc()
                        _log(f"Error: {str(e)}\n{tb}", debug)
                        return CallToolResult(
                            isError=True,
                            content=[TextContent(type="text", text=f"Error: {str(e)}")]
                        )
                
                return CallToolResult(
                    isError=True,
                    content=[TextContent(type="text", text=f"Tool not found: {tool_name}")]
                )
            
            # Run the MCP server
            print(f"\n🚀 Starting MCP server '{name}' with {transport} transport")
            try:
                if transport == "stdio":
                    mcp_server.run(transport="stdio")
                elif transport == "sse":
                    print(f"📡 Listening on http://{host}:{port}")
                    mcp_server.run(transport="sse", host=host, port=port)
                else:
                    raise ValueError(f"Unsupported transport: {transport}")
            except Exception as e:
                import traceback
                print(f"ERROR in run: {str(e)}\n{traceback.format_exc()}")
            
            return mcp_server
            
        return wrapper
    return decorator
