from langchain_openai import OpenAI
from langchain.agents import AgentType, Tool, initialize_agent
from auto_mcp import mcp_server
from dotenv import load_dotenv
import os
import sys

# Load environment variables from .env file
load_dotenv()

# Verify that the API key is available
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable is not set.")
    print("Please set it in your .env file or export it in your environment.")
    sys.exit(1)

def add_numbers_tool(query: str) -> str:
    """Add numbers provided in the query."""
    try:
        numbers = [int(num) for num in query.split() if num.isdigit()]
        if not numbers:
            return "No numbers found in the query."
        return f"The sum of {', '.join(map(str, numbers))} is {sum(numbers)}"
    except ValueError:
        return "Error: Could not parse integers from query."

def multiply_numbers_tool(query: str) -> str:
    """Multiply numbers provided in the query."""
    try:
        numbers = [int(num) for num in query.split() if num.isdigit()]
        if not numbers:
            return "No numbers found in the query."
        result = 1
        for num in numbers:
            result *= num
        return f"The product of {', '.join(map(str, numbers))} is {result}"
    except ValueError:
        return "Error: Could not parse integers from query."

@mcp_server(name="Math Operations Agent")
def simple_agent():
    """Create a simple agent that can perform math operations."""
    tools = [
        Tool(
            name="Addition Tool",
            func=add_numbers_tool,
            description="Useful for adding numbers. Provide text with integers to add."
        ),
        Tool(
            name="Multiplication Tool",
            func=multiply_numbers_tool,
            description="Useful for multiplying numbers. Provide text with integers to multiply."
        )
    ]
    
    # Create and return the agent executor
    return initialize_agent(
        tools,
        OpenAI(temperature=0),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

if __name__ == "__main__":
    simple_agent()
