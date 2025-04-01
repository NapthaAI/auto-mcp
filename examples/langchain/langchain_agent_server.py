import logging
import os
import sys
from dotenv import load_dotenv
from langchain_community.chat_models import ChatPerplexity
from langgraph.prebuilt import create_react_agent

from auto_mcp import langchain_mcp
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler('langchain-agent.log')  # Also log to a file
    ]
)

# Load environment variables from .env file
load_dotenv()

# Create logger for both modules to use
logger = logging.getLogger("langchain-agent")

# Set up API key
def setup_api_key():
    """Verify API key and return it if available"""
    perplexity_api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not perplexity_api_key:
        print("Error: PERPLEXITY_API_KEY environment variable is not set.")
        print("Please set it in your .env file or export it in your environment.")
        sys.exit(1)
    return perplexity_api_key

# Initialize your language model and agent
def initialize_agent():
    """Initialize Perplexity language model and create Langchain agent"""
    perplexity_api_key = setup_api_key()
    
    logger.info("Initializing Perplexity language model")
    model = ChatPerplexity(
        model="sonar",
        api_key=perplexity_api_key
    )

    logger.info("Creating Langchain agent")
    return create_react_agent(model, tools=[], debug=True)

# Global agent instance
agent = initialize_agent()

# Define input schema for the MCP server
class QueryInput(BaseModel):
    query: str
    
    class Config:
        extra = "forbid"

@langchain_mcp(
    input_schema=QueryInput,
    name="Perplexity Agent",
    description="Process queries using the Perplexity language model",
    debug=True
)
async def ask_pplx(query: str):
    """
    Process a query using the Langchain agent.
    
    Args:
        query: The user's question or instruction
        
    Returns:
        The agent's response as a string
    """
    logger.info(f"Received query: {query}")
    try:
        logger.debug("Invoking agent")
        # Use await to properly wait for the async result
        result = await agent.ainvoke({"messages": query})
        logger.info("Agent processed query successfully")
        logger.debug(f"Agent response: {result}")
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return f"Error processing query: {str(e)}"

if __name__ == "__main__":
    logger.info("Starting MCP server with stdio transport")
    ask_pplx()
    logger.info("MCP server stopped")
