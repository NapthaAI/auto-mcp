import logging
import os
import sys
from dotenv import load_dotenv
from langchain_community.chat_models import ChatPerplexity
from langgraph.prebuilt import create_react_agent
from auto_mcp import mcp_server

import asyncio

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

# Verify that the API key is available
perplexity_api_key = os.environ.get("PERPLEXITY_API_KEY")
if not perplexity_api_key:
    print("Error: PERPLEXITY_API_KEY environment variable is not set.")
    print("Please set it in your .env file or export it in your environment.")
    sys.exit(1)

logger = logging.getLogger("langchain-agent")

# Initialize your language model
logger.info("Initializing Perplexity language model")
model = ChatPerplexity(
    model="sonar",
    api_key=perplexity_api_key  # Explicitly pass the API key
)

# Create your Langchain agent
logger.info("Creating Langchain agent")
agent = create_react_agent(model, tools=[], debug=True)

# Initialize the MCP server
logger.info("Initializing MCP server")

@mcp_server(name="Perplexity Agent", as_tool=True, debug=True)
async def ask_pplx(query: str) -> str:
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
        return str(result)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return f"Error processing query: {str(e)}"

if __name__ == "__main__":
    logger.info("Starting MCP server with stdio transport")
    ask_pplx()
    logger.info("MCP server stopped")
