import sys
from crew import StockAnalysisCrew
from pydantic import BaseModel, ConfigDict

from auto_mcp import crewai_mcp

class QueryInput(BaseModel):
    query: str
    company_stock: str = "AMZN"
    # Replace old Config class with model_config for Pydantic V2
    model_config = ConfigDict(extra="forbid")
    

@crewai_mcp(name="financial_agent", input_schema=QueryInput)
def run():
    """
    Run the financial agent as an MCP server.
    When called through MCP, this function will receive the validated
    QueryInput object through the crewai_mcp wrapper.
    """
    # Since the MCP decorator handles the input passing, 
    # this function will execute the crew with the inputs provided by the MCP client
    crew = StockAnalysisCrew().crew()
    return crew

def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'query': 'What is last years revenue',
        'company_stock': 'AMZN',
    }
    try:
        StockAnalysisCrew().crew().train(n_iterations=int(sys.argv[1]), inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")
    
if __name__ == "__main__":
    print("## Welcome to Stock Analysis Crew")
    print('-------------------------------')
    result = run()
    print("\n\n########################")
    print("## Here is the Report")
    print("########################\n")
    print(result)
