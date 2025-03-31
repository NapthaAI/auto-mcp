import inspect
import asyncio
import functools
import sys
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from mcp.server.fastmcp import FastMCP
from mcp.types import Tool as MCPTool
from mcp.types import CallToolResult, TextContent

class LangChainMCPAdapter:
    """Adapter that converts LangChain agents to MCP servers."""
    
    def __init__(
        self,
        agent_factory: Callable,
        name: Optional[str] = None,
        transport: Literal["stdio", "sse"] = "stdio",
        host: str = "localhost",
        port: int = 8000,
        is_tool_function: bool = False,
    ):
        self.agent_factory = agent_factory
        self.agent = None
        self.transport = transport
        self.host = host
        self.port = port
        self.is_tool_function = is_tool_function
        
        # Store function name and server name for alternate tool naming
        self.function_name = agent_factory.__name__
        server_name = name or f"LangChain-{agent_factory.__name__}"
        self.server_name = server_name
        
        # Generate alternate name for the tool based on server name
        self.alt_tool_name = self.server_name.split()[0].lower() if " " in self.server_name else server_name.lower()
        
        self.mcp_server = FastMCP(server_name)
        self._setup_mcp_server()
    
    async def _run_tool(self, tool, arguments: Dict[str, Any]) -> Any:
        """Run a tool with the given arguments, handling both sync and async tools."""
        try:
            if hasattr(tool, 'invoke'):
                # Structured tool
                if asyncio.iscoroutinefunction(tool.invoke):
                    return await tool.invoke(arguments)
                else:
                    return tool.invoke(arguments)
            elif hasattr(tool, 'func') and callable(tool.func):
                # Simple function-based tool
                if asyncio.iscoroutinefunction(tool.func):
                    if len(arguments) == 1 and "query" in arguments:
                        return await tool.func(arguments["query"])
                    else:
                        return await tool.func(**arguments)
                else:
                    if len(arguments) == 1 and "query" in arguments:
                        return tool.func(arguments["query"])
                    else:
                        return tool.func(**arguments)
            elif hasattr(tool, 'run') and callable(tool.run):
                # Legacy LangChain tool
                if asyncio.iscoroutinefunction(tool.run):
                    return await tool.run(str(arguments))
                else:
                    return tool.run(str(arguments))
            else:
                raise ValueError(f"Unsupported tool type: {type(tool)}")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            raise Exception(f"Error executing tool {tool.name}: {str(e)}\n{tb}")
    
    def _extract_tool_parameters(self, tool) -> Dict[str, Dict[str, Any]]:
        """Extract parameter information from a LangChain tool."""
        params = {}
        
        # Try to extract schema from args_schema if available
        if hasattr(tool, 'args_schema') and tool.args_schema:
            schema_cls = tool.args_schema
            if hasattr(schema_cls, "__fields__"):  # Pydantic v1
                for field_name, field in schema_cls.__fields__.items():
                    param_type = "string"  # Default type
                    if hasattr(field.type_, "__name__"):
                        param_type = field.type_.__name__.lower()
                    elif hasattr(field.type_, "_name"):
                        param_type = field.type_._name.lower()
                    
                    # Map Python types to JSON Schema types
                    type_mapping = {
                        "int": "integer",
                        "float": "number",
                        "str": "string",
                        "bool": "boolean",
                        "list": "array",
                        "dict": "object",
                    }
                    param_type = type_mapping.get(param_type, "string")
                    
                    params[field_name] = {
                        "type": param_type,
                        "description": field.description or "",
                        "required": field.required
                    }
            elif hasattr(schema_cls, "model_fields"):  # Pydantic v2
                for field_name, field in schema_cls.model_fields.items():
                    param_type = "string"  # Default type
                    if hasattr(field.annotation, "__name__"):
                        param_type = field.annotation.__name__.lower()
                    
                    # Map Python types to JSON Schema types
                    type_mapping = {
                        "int": "integer",
                        "float": "number",
                        "str": "string",
                        "bool": "boolean",
                        "list": "array",
                        "dict": "object",
                    }
                    param_type = type_mapping.get(param_type, "string")
                    
                    params[field_name] = {
                        "type": param_type,
                        "description": field.description if hasattr(field, "description") else "",
                        "required": not field.is_optional() if hasattr(field, "is_optional") else True
                    }
        # If it's a function-based tool, extract parameters from function signature
        elif hasattr(tool, 'func') and callable(tool.func):
            sig = inspect.signature(tool.func)
            for param_name, param in sig.parameters.items():
                if param_name == "self" or param_name == "cls" or param_name == "ctx":
                    continue
                
                param_type = "string"  # Default
                if param.annotation != inspect.Parameter.empty:
                    type_name = param.annotation.__name__ if hasattr(param.annotation, "__name__") else str(param.annotation)
                    type_mapping = {
                        "int": "integer",
                        "float": "number",
                        "str": "string",
                        "bool": "boolean",
                        "list": "array",
                        "dict": "object",
                    }
                    param_type = type_mapping.get(type_name.lower(), "string")
                
                params[param_name] = {
                    "type": param_type,
                    "description": "",
                    "required": param.default == inspect.Parameter.empty
                }
        # Otherwise create a generic "query" parameter
        else:
            params["query"] = {
                "type": "string",
                "description": f"Input for {tool.name}",
                "required": True
            }
        
        return params
    
    def _extract_function_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Extract parameter information from a function."""
        params = {}
        sig = inspect.signature(self.agent_factory)
        for param_name, param in sig.parameters.items():
            if param_name == "self" or param_name == "cls" or param_name == "ctx":
                continue
            
            param_type = "string"  # Default
            if param.annotation != inspect.Parameter.empty:
                type_name = param.annotation.__name__ if hasattr(param.annotation, "__name__") else str(param.annotation)
                type_mapping = {
                    "int": "integer",
                    "float": "number",
                    "str": "string",
                    "bool": "boolean",
                    "list": "array",
                    "dict": "object",
                }
                param_type = type_mapping.get(type_name.lower(), "string")
            
            params[param_name] = {
                "type": param_type,
                "description": "",
                "required": param.default == inspect.Parameter.empty
            }
        return params
    
    async def _handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle the execution of a tool call."""
        print(f"DEBUG: Tool call received - name: {name}, arguments: {arguments}")
        
        # Special handling for direct tool call with server name matches
        if name == self.server_name and isinstance(arguments, str):
            print(f"DEBUG: Direct call pattern detected with string argument for {self.server_name}")
            processed_args = {"query": arguments}
            
            try:
                result = await self._smart_execute_function(processed_args)
                return CallToolResult(
                    isError=False,
                    content=[TextContent(type="text", text=str(result))]
                )
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                return CallToolResult(
                    isError=True,
                    content=[TextContent(type="text", text=f"Error: {str(e)}\n{tb}")]
                )
        
        # Special handling for nested call_tool pattern
        if name == "call_tool" and isinstance(arguments, dict) and "name" in arguments and "arguments" in arguments:
            nested_name = arguments["name"]
            nested_args = arguments["arguments"]
            print(f"DEBUG: Detected nested call_tool pattern - name: {nested_name}, args: {nested_args}")
            
            if nested_name == self.server_name and isinstance(nested_args, str):
                print(f"DEBUG: Special case - {self.server_name} with string argument")
                processed_args = {"query": nested_args}
                
                try:
                    result = await self._smart_execute_function(processed_args)
                    return CallToolResult(
                        isError=False,
                        content=[TextContent(type="text", text=str(result))]
                    )
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    return CallToolResult(
                        isError=True,
                        content=[TextContent(type="text", text=f"Error: {str(e)}\n{tb}")]
                    )
            
            return await self._handle_tool_call(nested_name, nested_args)
        
        # Handle tool function mode
        if self.is_tool_function:
            func_name = self.function_name
            alt_names = [
                self.alt_tool_name,
                self.server_name.lower().replace(" ", "-"),
                self.server_name.lower().replace(" ", "_"),
                self.server_name.lower().replace(" ", ""),
                self.server_name,
                self.server_name.lower(),
            ]
            
            normalized_name = name.lower().strip()
            print(f"DEBUG: Checking tool name '{normalized_name}' against alternatives:")
            print(f"DEBUG: Function name: {func_name.lower()}")
            print(f"DEBUG: Alt names: {[n.lower() for n in alt_names]}")
            
            if (normalized_name == func_name.lower() or 
                normalized_name in [n.lower() for n in alt_names] or
                normalized_name == self.server_name.lower()):
                
                print(f"DEBUG: Tool name matched! Processing request for tool: {func_name}")
                
                try:
                    param_info = self._extract_function_parameters()
                    processed_args = arguments
                    
                    if "query" in param_info:
                        if isinstance(arguments, str):
                            processed_args = {"query": arguments}
                            print(f"DEBUG: Wrapped string as query: {processed_args}")
                        elif isinstance(arguments, dict) and "query" not in arguments:
                            if len(arguments) > 0:
                                first_key = next(iter(arguments.keys()))
                                first_value = arguments[first_key]
                                if isinstance(first_value, str):
                                    processed_args = {"query": first_value}
                                    print(f"DEBUG: Extracted query from dict value: {processed_args}")
                                else:
                                    processed_args = {"query": str(arguments)}
                                    print(f"DEBUG: Using entire dict as query: {processed_args}")
                    
                    print(f"DEBUG: Final arguments after processing: {processed_args}")
                    result = await self._smart_execute_function(processed_args)
                    print(f"DEBUG: Function result: {result}")
                    
                    return CallToolResult(
                        isError=False,
                        content=[TextContent(type="text", text=str(result))]
                    )
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    print(f"ERROR: Exception when calling function: {str(e)}\n{tb}")
                    return CallToolResult(
                        isError=True,
                        content=[TextContent(type="text", text=f"Error: {str(e)}\n{tb}")]
                    )
        
        # Standard agent tool handling
        if not self.is_tool_function:
            if self.agent is None:
                try:
                    self.agent = self.agent_factory()
                except TypeError:
                    return CallToolResult(
                        isError=True,
                        content=[TextContent(type="text", text="Agent factory requires arguments and cannot be initialized automatically.")]
                    )
            
            if hasattr(self.agent, 'tools'):
                tools_list = self.agent.tools
            elif hasattr(self.agent, 'agent') and hasattr(self.agent.agent, 'tools'):
                tools_list = self.agent.agent.tools
            elif hasattr(self.agent, 'toolkit') and hasattr(self.agent.toolkit, 'get_tools'):
                tools_list = self.agent.toolkit.get_tools()
            else:
                return CallToolResult(
                    isError=True,
                    content=[TextContent(type="text", text="Could not find tools in the agent")]
                )
            
            matching_tools = [t for t in tools_list if t.name == name]
            if not matching_tools:
                return CallToolResult(
                    isError=True,
                    content=[TextContent(type="text", text=f"Tool not found: {name}")]
                )
            
            tool = matching_tools[0]
            try:
                result = await self._run_tool(tool, arguments)
                return CallToolResult(
                    isError=False,
                    content=[TextContent(type="text", text=str(result))]
                )
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                return CallToolResult(
                    isError=True,
                    content=[TextContent(type="text", text=f"Error: {str(e)}\n{tb}")]
                )
        
        return CallToolResult(
            isError=True,
            content=[TextContent(type="text", text=f"Tool not found: {name}")]
        )
    
    def _setup_mcp_server(self):
        """Set up the MCP server with tools from the agent."""
        @self.mcp_server.tool()
        def list_tools() -> List[MCPTool]:
            mcp_tools = []
            
            if self.is_tool_function:
                tool_params = self._extract_function_parameters()
                func_name = self.function_name
                doc = self.agent_factory.__doc__ or f"Execute {func_name}"
                
                primary_tool = MCPTool(
                    name=func_name,
                    description=doc,
                    inputSchema=tool_params
                )
                mcp_tools.append(primary_tool)
                
                alt_names = [
                    self.alt_tool_name,
                    self.server_name.lower().replace(" ", "-"),
                    self.server_name.lower().replace(" ", "_"),
                    self.server_name.lower().replace(" ", ""),
                    self.server_name,
                    self.server_name.lower(),
                ]
                if self.server_name.lower() != func_name.lower() and self.server_name not in alt_names:
                    alt_names.append(self.server_name)
                
                if ' ' in self.server_name:
                    first_word = self.server_name.split(' ')[0].lower()
                    if first_word not in alt_names:
                        alt_names.append(first_word)
                
                for alt_name in set(alt_names):
                    if alt_name != func_name.lower() and alt_name not in [t.name.lower() for t in mcp_tools]:
                        if not alt_name.strip():
                            continue
                        alt_tool = MCPTool(
                            name=alt_name,
                            description=f"Alternative name for {func_name}: {doc}",
                            inputSchema=tool_params
                        )
                        mcp_tools.append(alt_tool)
                
                print(f"DEBUG: Registered {len(mcp_tools)} tools: {[t.name for t in mcp_tools]}")
                return mcp_tools
            
            if self.agent is None:
                try:
                    self.agent = self.agent_factory()
                except TypeError:
                    print("Warning: Agent factory requires arguments and cannot be initialized automatically.")
                    return []
            
            if hasattr(self.agent, 'tools'):
                tools_list = self.agent.tools
            elif hasattr(self.agent, 'agent') and hasattr(self.agent.agent, 'tools'):
                tools_list = self.agent.agent.tools
            elif hasattr(self.agent, 'toolkit') and hasattr(self.agent.toolkit, 'get_tools'):
                tools_list = self.agent.toolkit.get_tools()
            else:
                print("Warning: Could not find tools in the agent.")
                tools_list = []
            
            for lc_tool in tools_list:
                tool_params = self._extract_tool_parameters(lc_tool)
                mcp_tool = MCPTool(
                    name=lc_tool.name,
                    description=lc_tool.description,
                    inputSchema=tool_params
                )
                mcp_tools.append(mcp_tool)
            return mcp_tools
        
        @self.mcp_server.tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            return await self._handle_tool_call(name, arguments)
    
    async def _smart_execute_function(self, arguments):
        """Intelligently execute a function handling async and sync cases automatically."""
        if asyncio.iscoroutinefunction(self.agent_factory):
            print(f"DEBUG: Calling async function directly: {self.function_name}")
            return await self.agent_factory(**arguments)
        
        try:
            function_code = ""
            try:
                function_code = inspect.getsource(self.agent_factory)
            except Exception:
                pass
            
            contains_async_code = any(keyword in function_code for keyword in 
                                ['await ', 'async ', 'ainvoke', 'acall', 'arun'])
            print(f"DEBUG: Executing function: {self.function_name}")
            result = self.agent_factory(**arguments)
            
            # Check if result is already a coroutine
            if asyncio.iscoroutine(result):
                print("DEBUG: Function returned a coroutine, awaiting it")
                return await result
            
            # Handle the case where the result contains a coroutine that wasn't awaited
            result_str = str(result)
            if '<coroutine object' in result_str:
                print("DEBUG: Detected unwrapped coroutine in result string")
                
                # Re-execute the function and properly await the coroutine
                async def execute_with_await():
                    fresh_result = self.agent_factory(**arguments)
                    if asyncio.iscoroutine(fresh_result):
                        return await fresh_result
                    
                    if 'agent.ainvoke(' in function_code or '.ainvoke(' in function_code:
                        from langchain_core.runnables.base import Runnable
                        if isinstance(fresh_result, Runnable):
                            # Handle LangChain Runnable
                            query = arguments.get("query", "")
                            return await fresh_result.ainvoke({"messages": query})
                    
                    return fresh_result
                
                return await execute_with_await()
            
            if contains_async_code or '.ainvoke(' in function_code:
                print(f"DEBUG: Function {self.function_name} appears to use async code")
                if any(pattern in result_str for pattern in ['coroutine', 'awaitable', '_call_with_retry']):
                    print("DEBUG: Detected unresolved coroutine in result, handling internally")
                    
                    async def execute_async():
                        fresh_result = self.agent_factory(**arguments)
                        if asyncio.iscoroutine(fresh_result):
                            return await fresh_result
                        return fresh_result
                    
                    return await execute_async()
                
                if "messages" in result_str and ("ainvoke" in function_code or "agent.invoke" in function_code):
                    print("DEBUG: Detected LangChain agent invoke pattern")
                    import re
                    agent_pattern = re.search(r'(\w+)\.ainvoke\(', function_code)
                    if agent_pattern:
                        agent_var = agent_pattern.group(1)
                        print(f"DEBUG: Detected agent variable: {agent_var}")
                        async def handle_langchain_agent():
                            async def wrapper():
                                modified_code = function_code.replace(f'{agent_var}.ainvoke(', f'await {agent_var}.ainvoke(')
                                modified_code = modified_code.replace(f'def {self.function_name}', f'async def {self.function_name}_async')
                                local_context = {**globals(), agent_var: result}
                                exec(modified_code, globals(), local_context)
                                async_func = local_context[f"{self.function_name}_async"]
                                return await async_func(**arguments)
                            return await wrapper()
                        
                        try:
                            return await handle_langchain_agent()
                        except Exception as e:
                            print(f"DEBUG: Specialized handler failed: {str(e)}")
            
            return result
        except Exception as e:
            import traceback
            print(f"ERROR in _smart_execute_function: {str(e)}\n{traceback.format_exc()}")
            raise

    def run(self):
        """Run the MCP server."""
        print(f"\n🚀 Starting LangChain MCP server with {self.transport} transport")
        print(f"💡 This server will expose your LangChain agent's tools via the Model Context Protocol")
        print(f"🔌 Compatible with Claude Desktop, Cursor, and other MCP clients\n")
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

def mcp_server(func=None, 
    *, 
    name=None, 
    transport: Literal["stdio", "sse"] = "stdio",
    host: str = "localhost",
    port: int = 8000,                        
    debug: bool = False,            
    as_tool: bool = False):
    """
    Decorator that converts a LangChain agent factory into an MCP server.
    
    Args:
        func: The function to decorate
        name: Optional name for the MCP server
        transport: The transport protocol to use ("stdio" or "sse")
        host: Host to bind to when using SSE transport
        port: Port to bind to when using SSE transport
        debug: Whether to print debug information
        as_tool: Whether to expose the function directly as a tool (useful for functions with required parameters)
    
    Example:
        @mcp_server
        def my_agent():
            # Create and return a LangChain agent
            return initialize_agent(...)
    
        # Or with custom options: @mcp_server(name="MyCustomServer", transport="sse", port=9000)
        @mcp_server(name="MyCustomServer", transport="sse", port=9000)
        def my_agent():
            return initialize_agent(...)
            
        # As a tool function with required parameters:
        @mcp_server(as_tool=True)
        def ask_question(query: str):
            # Process the query and return a result
            return process_query(query)
            
        # Functions with async code work without special handling:
        @mcp_server(as_tool=True)
        def ask_pplx(query: str):
            # The decorator automatically detects and handles the async call
            return agent.ainvoke({"messages": query})
    """
    def decorator(agent_factory):
        @functools.wraps(agent_factory)
        def wrapper(*args, **kwargs):
            if debug:
                print(f"DEBUG: Initializing MCP server for {agent_factory.__name__}")
                print(f"DEBUG: as_tool={as_tool}, transport={transport}")
            
            adapter = LangChainMCPAdapter(
                agent_factory,
                name=name,
                transport=transport,
                host=host,
                port=port,
                is_tool_function=as_tool,
            )
            
            if '--direct' in sys.argv or '-d' in sys.argv:
                if as_tool and len(args) == 0 and not kwargs:
                    print(f"\n🛠️ Function {agent_factory.__name__} requires parameters.")
                    print(f"Run with: python script.py --direct {agent_factory.__name__}_param=value")
                    return adapter
                    
                if as_tool:
                    if not args and not kwargs:
                        for arg in sys.argv:
                            if "=" in arg:
                                k, v = arg.split("=", 1)
                                kwargs[k] = v
                    
                    print(f"\n🛠️ Running function directly: {agent_factory.__name__}")
                    result = agent_factory(*args, **kwargs)
                    print(f"\n📝 Result: {result}")
                else:
                    try:
                        agent = agent_factory(*args, **kwargs)
                        print("\n🛠️ Running agent directly (not as MCP server)")
                        print("💬 Example query: What is the sum of 5 10 and 15?")
                        
                        while True:
                            try:
                                query = input("\n🔍 Enter query (or 'exit' to quit): ")
                                if query.lower() in ['exit', 'quit', 'q']:
                                    break
                                
                                print("\n⚙️ Agent processing...")
                                if hasattr(agent, 'run'):
                                    result = agent.run(query)
                                    print(f"📝 Result: {result}")
                                elif hasattr(agent, 'invoke'):
                                    result = agent.invoke(query)
                                    print(f"📝 Result: {result}")
                                else:
                                    print("\n❌ Error: Agent doesn't have run or invoke method")
                            except Exception as e:
                                print(f"\n❌ Error: {str(e)}")
                    except TypeError as e:
                        print(f"\n❌ Error initializing agent: {str(e)}")
                        print("The agent factory requires parameters. Please provide them.")
            else:
                adapter.run()
                
            return adapter
        
        wrapper.original = agent_factory
        
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)

if __name__ == "__main__":
    print("This module provides the mcp_server decorator to convert LangChain agents into MCP servers.")
    print("Import and use it in your own scripts with: from auto_mcp import mcp_server")