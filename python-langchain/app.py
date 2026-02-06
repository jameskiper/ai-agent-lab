import os
from datetime import datetime
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
  """
  Evaluates a mathematical expression provided as a string.
  Uses Python's eval() for demonstration purposes ONLY.
  Returns the result as a string or an error message.
  """
  try:
    # Disable builtins for basic safety (demo purposes)
    # If your linter/IDE complains, you can use a safer approach:
    allowed_names = {}
    result = eval(expression, {"__builtins__": None}, allowed_names)
    return str(result)
  except Exception as e:
    return f"Error: {str(e)}"





@tool
def get_current_time() -> str:
  """
  Returns the current date and time as a formatted string.
  Uses datetime.now().strftime("%Y-%m-%d %H:%M:%S").
  """
  return datetime.now().strftime("%Y-%m-%d %H:%M:%S")



def main():
  load_dotenv()
  print(" Starting application...")

  # Optional GitHub token (used later for GitHub-hosted models)
  github_token = os.getenv("GITHUB_TOKEN")

  if not github_token:
    print(" GITHUB_TOKEN not found in environment variables.")
    print(" Please set your GitHub token in a .env file or as an environment variable.")
    print("   Example .env entry: GITHUB_TOKEN=your_token_here")
    return

  # Initialize ChatOpenAI model using GitHub-hosted endpoint
  llm = ChatOpenAI(
    model="openai/gpt-4o",
    temperature=0,
    base_url="https://models.github.ai/inference",
    api_key=github_token,
  )

  print("ChatOpenAI model initialized successfully!")

  # Tool list (Prompt 14)
  tools = [calculator, get_current_time]
  
  # Test the get_current_time tool directly
  print("Testing get_current_time tool:", get_current_time.invoke({}))
  
  # Create an agent that can use the tools (Prompt 14)
  agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant. Use the available tools to answer questions."
  )
  
  # Test query (Prompt 14: With tool)
  query = "What time is it right now?"



  try:
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    # Extract the final AI response from the messages
    final_response = result["messages"][-1].content
    print("Response:", final_response)
  except Exception as e:
    print(" Error during model invocation:", str(e))
    print(" Please check that your GITHUB_TOKEN is valid and has access to https://models.github.ai/inference.")
    print("   If you are using a personal access token, ensure it has the correct scopes.")
    return

if __name__ == "__main__":
  main()