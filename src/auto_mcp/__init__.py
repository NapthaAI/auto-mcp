import inspect
import asyncio
import functools
import sys
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union

from mcp.server.fastmcp import FastMCP
from mcp.types import Tool as MCPTool
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, create_model


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
            print(f"DEBUG: {message}")
    
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
        print(f"\n🚀 Starting MCP server '{self.name}' with {self.transport} transport")
        try:
            if self.transport == "stdio":
                self.mcp_server.run(transport="stdio")
            elif self.transport == "sse":
                print(f"📡 Listening on http://{self.host}:{self.port}")
                self.mcp_server.run(transport="sse", host=self.host, port=self.port)
            else:
                raise ValueError(f"Unsupported transport: {self.transport}")
        except Exception as e:
            import traceback
            print(f"ERROR in run: {str(e)}\n{traceback.format_exc()}")


class LangChainAdapter(BaseMCPAdapter):
    """Adapter for LangChain agents."""
    
    def __init__(
        self,
        agent_factory: Callable,
        name: str,
        input_schema: Type[BaseModel],
        transport: Literal["stdio", "sse"] = "stdio",
        host: str = "localhost",
        port: int = 8000,
        description: str = None,
        debug: bool = False,
    ):
        self.agent_factory = agent_factory
        self.agent = None
        super().__init__(name, input_schema, transport, host, port, description, debug)
    
    async def _handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle tool calls for LangChain agents."""
        self._log(f"Tool call: {name} with arguments: {arguments}")
        
        # Handle nested call_tool pattern
        if name == "call_tool" and isinstance(arguments, dict) and "name" in arguments and "arguments" in arguments:
            nested_name = arguments["name"]
            nested_args = arguments["arguments"]
            self._log(f"Nested call_tool - name: {nested_name}")
            return await self._handle_tool_call(nested_name, nested_args)
        
        # Handle call to our main tool
        if name == self.name:
            try:
                # Convert string arguments to proper input schema if needed
                if isinstance(arguments, str):
                    arguments = {"query": arguments}
                
                # Validate input against schema
                input_data = self.input_schema(**arguments)
                
                # Call the agent factory with validated input
                if self.agent is None:
                    if asyncio.iscoroutinefunction(self.agent_factory):
                        self.agent = await self.agent_factory(**input_data.model_dump())
                    else:
                        self.agent = self.agent_factory(**input_data.model_dump())
                
                # Process input with the agent
                if hasattr(self.agent, "invoke"):
                    if asyncio.iscoroutinefunction(self.agent.invoke):
                        result = await self.agent.invoke(input_data.model_dump())
                    else:
                        result = self.agent.invoke(input_data.model_dump())
                elif hasattr(self.agent, "run"):
                    if asyncio.iscoroutinefunction(self.agent.run):
                        result = await self.agent.run(str(input_data.model_dump()))
                    else:
                        result = self.agent.run(str(input_data.model_dump()))
                else:
                    result = str(self.agent)
                
                return CallToolResult(
                    isError=False,
                    content=[TextContent(type="text", text=str(result))]
                )
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self._log(f"Error: {str(e)}\n{tb}")
                return CallToolResult(
                    isError=True,
                    content=[TextContent(type="text", text=f"Error: {str(e)}")]
                )
        
        return CallToolResult(
            isError=True,
            content=[TextContent(type="text", text=f"Tool not found: {name}")]
        )


class CrewAIAdapter(BaseMCPAdapter):
    """Adapter for CrewAI crews."""
    
    def __init__(
        self,
        crew_factory: Callable,
        name: str,
        input_schema: Type[BaseModel],
        transport: Literal["stdio", "sse"] = "stdio",
        host: str = "localhost",
        port: int = 8000,
        description: str = None,
        debug: bool = False,
    ):
        self.crew_factory = crew_factory
        self.crew = None
        super().__init__(name, input_schema, transport, host, port, description, debug)
    
    async def _handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle tool calls for CrewAI crews."""
        self._log(f"Tool call: {name} with arguments: {arguments}")
        
        # Handle nested call_tool pattern
        if name == "call_tool" and isinstance(arguments, dict) and "name" in arguments and "arguments" in arguments:
            nested_name = arguments["name"]
            nested_args = arguments["arguments"]
            self._log(f"Nested call_tool - name: {nested_name}")
            return await self._handle_tool_call(nested_name, nested_args)
        
        # Handle call to our main tool
        if name == self.name:
            try:
                # Convert string arguments to proper input schema if needed
                if isinstance(arguments, str):
                    arguments = {"query": arguments}
                
                # Validate input against schema
                input_data = self.input_schema(**arguments)
                
                # Initialize the crew if needed
                if self.crew is None:
                    self.crew = self.crew_factory()
                
                # Run the crew
                result = self.crew.kickoff(inputs=input_data.model_dump())
                
                # Try to convert result to JSON if it's a Pydantic model
                if hasattr(result, "model_dump_json"):
                    result_str = result.model_dump_json()
                else:
                    result_str = str(result)
                
                return CallToolResult(
                    isError=False,
                    content=[TextContent(type="text", text=result_str)]
                )
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self._log(f"Error: {str(e)}\n{tb}")
                return CallToolResult(
                    isError=True,
                    content=[TextContent(type="text", text=f"Error: {str(e)}")]
                )
        
        return CallToolResult(
            isError=True,
            content=[TextContent(type="text", text=f"Tool not found: {name}")]
        )


class CamelAIAdapter(BaseMCPAdapter):
    """Adapter for CamelAI agents."""
    
    def __init__(
        self,
        agent_factory: Callable,
        name: str,
        input_schema: Type[BaseModel],
        transport: Literal["stdio", "sse"] = "stdio",
        host: str = "localhost",
        port: int = 8000,
        description: str = None,
        debug: bool = False,
    ):
        self.agent_factory = agent_factory
        self.agent = None
        super().__init__(name, input_schema, transport, host, port, description, debug)
    
    async def _handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle tool calls for CamelAI agents."""
        self._log(f"Tool call: {name} with arguments: {arguments}")
        
        # Handle nested call_tool pattern
        if name == "call_tool" and isinstance(arguments, dict) and "name" in arguments and "arguments" in arguments:
            nested_name = arguments["name"]
            nested_args = arguments["arguments"]
            self._log(f"Nested call_tool - name: {nested_name}")
            return await self._handle_tool_call(nested_name, nested_args)
        
        # Handle call to our main tool
        if name == self.name:
            try:
                # Convert string arguments to proper input schema if needed
                if isinstance(arguments, str):
                    arguments = {"query": arguments}
                
                # Validate input against schema
                input_data = self.input_schema(**arguments)
                
                # Initialize the agent if needed
                if self.agent is None:
                    self.agent = self.agent_factory()
                
                # Call the appropriate method based on what's available
                if hasattr(self.agent, "run"):
                    result = self.agent.run(**input_data.model_dump())
                elif hasattr(self.agent, "chat"):
                    result = self.agent.chat(**input_data.model_dump())
                else:
                    raise ValueError("CamelAI agent has no run or chat method")
                
                return CallToolResult(
                    isError=False,
                    content=[TextContent(type="text", text=str(result))]
                )
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self._log(f"Error: {str(e)}\n{tb}")
                return CallToolResult(
                    isError=True,
                    content=[TextContent(type="text", text=f"Error: {str(e)}")]
                )
        
        return CallToolResult(
            isError=True,
            content=[TextContent(type="text", text=f"Tool not found: {name}")]
        )


# Framework-specific decorators

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
            
            adapter = LangChainAdapter(
                agent_factory=func,
                name=name,
                input_schema=input_schema,
                transport=transport,
                host=host,
                port=port,
                description=description or func.__doc__,
                debug=debug,
            )
            adapter.run()
            return adapter
            
        return wrapper
    return decorator


def crewai_mcp(
    input_schema: Type[BaseModel],
    name: str,
    description: str = None,
    transport: Literal["stdio", "sse"] = "stdio",
    host: str = "localhost",
    port: int = 8000,
    debug: bool = False,
):
    """
    Decorator for converting CrewAI crews to MCP servers.
    
    Args:
        input_schema: Pydantic model defining the input schema
        name: Name for the MCP server
        description: Description of the server
        transport: Transport protocol ("stdio" or "sse")
        host: Host for SSE transport
        port: Port for SSE transport
        debug: Enable debug logging
    
    Example:
        class MarketingInput(BaseModel):
            customer_domain: str
            project_description: str
        
        @crewai_mcp(input_schema=MarketingInput, name="Marketing Crew")
        def marketing_crew():
            return MarketingPostsCrew().crew()
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if args or kwargs:
                return func(*args, **kwargs)
            
            adapter = CrewAIAdapter(
                crew_factory=func,
                name=name,
                input_schema=input_schema,
                transport=transport,
                host=host,
                port=port,
                description=description or func.__doc__,
                debug=debug,
            )
            adapter.run()
            return adapter
            
        return wrapper
    return decorator


def camelai_mcp(
    input_schema: Type[BaseModel],
    name: str,
    description: str = None,
    transport: Literal["stdio", "sse"] = "stdio",
    host: str = "localhost",
    port: int = 8000,
    debug: bool = False,
):
    """
    Decorator for converting CamelAI agents to MCP servers.
    
    Args:
        input_schema: Pydantic model defining the input schema
        name: Name for the MCP server
        description: Description of the server
        transport: Transport protocol ("stdio" or "sse")
        host: Host for SSE transport
        port: Port for SSE transport
        debug: Enable debug logging
    
    Example:
        class ChatInput(BaseModel):
            message: str
        
        @camelai_mcp(input_schema=ChatInput, name="Customer Support")
        def support_agent():
            return CamelAgent()
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if args or kwargs:
                return func(*args, **kwargs)
            
            adapter = CamelAIAdapter(
                agent_factory=func,
                name=name,
                input_schema=input_schema,
                transport=transport,
                host=host,
                port=port,
                description=description or func.__doc__,
                debug=debug,
            )
            adapter.run()
            return adapter
            
        return wrapper
    return decorator
