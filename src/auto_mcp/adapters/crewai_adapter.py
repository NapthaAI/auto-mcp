import functools
import sys
import contextlib
import io
import warnings
from typing import Any, Callable, Dict, List, Literal, Type

from pydantic import BaseModel

from ..import BaseMCPAdapter
from mcp.types import CallToolResult, TextContent

# Helper class to suppress stdout/stderr
class NullWriter:
    def write(self, *args, **kwargs):
        pass
    
    def flush(self, *args, **kwargs):
        pass

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
            # Save original stdout and stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            try:
                # Convert string arguments to proper input schema if needed
                if isinstance(arguments, str):
                    arguments = {"query": arguments}
                
                # Validate input against schema
                input_data = self.input_schema(**arguments)
                
                # Initialize the crew if needed
                if self.crew is None:
                    # Redirect stdout during crew initialization
                    with contextlib.redirect_stdout(io.StringIO()):
                        with contextlib.redirect_stderr(io.StringIO()):
                            self.crew = self.crew_factory()
                
                # Run the crew with stdout and stderr redirected
                # This prevents interference with the Stdio transport
                stdout_buffer = io.StringIO()
                stderr_buffer = io.StringIO()
                
                with contextlib.redirect_stdout(stdout_buffer):
                    with contextlib.redirect_stderr(stderr_buffer):
                        # Also temporarily replace sys.stdout and sys.stderr for libraries
                        # that might bypass contextlib redirection
                        sys.stdout = NullWriter()
                        sys.stderr = NullWriter()
                        
                        try:
                            result = self.crew.kickoff(inputs=input_data.model_dump())
                        finally:
                            # Restore sys stdout/stderr within the context
                            sys.stdout = original_stdout
                            sys.stderr = original_stderr
                
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
            finally:
                # Ensure stdout and stderr are restored
                sys.stdout = original_stdout
                sys.stderr = original_stderr
        
        return CallToolResult(
            isError=True,
            content=[TextContent(type="text", text=f"Tool not found: {name}")]
        )

# Keep the existing decorator for backward compatibility
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
            
            # Apply warning filters
            warnings.filterwarnings("ignore")
            
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

# Add the new function-based approach
def crewai_to_mcp_tool(
    crewai_class: Any,
    name: str,
    description: str,
    input_schema: Type[BaseModel],
):
    """
    Convert a CrewAI class to an MCP tool.

    Args:
        crewai_class: The CrewAI class to convert
        name: The name of the tool
        description: The description of the tool
        input_schema: The Pydantic model class defining the input schema
    """
    # Define a typed function using the input schema
    def run_tool(input_data: BaseModel):
        try:
            # Ensure input_data is of the correct type
            if not isinstance(input_data, input_schema):
                # If received as dict or other format, convert to our schema
                if isinstance(input_data, dict):
                    input_data = input_schema(**input_data)
                else:
                    raise TypeError(f"Expected {input_schema.__name__}, got {type(input_data).__name__}")
                
            # Suppress standard output during execution
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()):
                    # Create and run the crew
                    crew = crewai_class().crew()
                    result = crew.kickoff(inputs=input_data.model_dump())
            
            # Return the result (MCP will handle the conversion)
            return result
            
        except Exception as e:
            import traceback
            error_msg = f"Error in crewai_to_mcp_tool: {str(e)}\n{traceback.format_exc()}"
            print(error_msg, file=sys.__stderr__)
            raise
    
    # Set metadata on the function
    run_tool.__name__ = name
    run_tool.__doc__ = description
    run_tool.__annotations__ = {"input_data": input_schema}
    
    # Return the function to be used as a tool
    return run_tool