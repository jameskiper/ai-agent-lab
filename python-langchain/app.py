import os
from datetime import datetime
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# @tool
# def calculator(expression: str) -> str:
#   """
#   Evaluates a mathematical expression provided as a string.
#   Uses Python's eval() for demonstration purposes ONLY.
#   Returns the result as a string or an error message.
#   """
#   try:
#     # Disable builtins for basic safety (demo purposes)
#     # If your linter/IDE complains, you can use a safer approach:
#     allowed_names = {}
#     result = eval(expression, {"__builtins__": None}, allowed_names)
#     return str(result)
#   except Exception as e:
#     return f"Error: {str(e)}"

# @tool
# def get_current_time() -> str:
#   """
#   Returns the current date and time as a formatted string.
#   Uses datetime.now().strftime("%Y-%m-%d %H:%M:%S").
#   """
#   return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
  tools = [get_current_time]

  # Test the get_current_time tool directly
  print("Testing get_current_time tool:", get_current_time.invoke({}))

  # Create an agent that can use the tools (Prompt 14)
  agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant. Use the available tools to answer questions."
  )

  # Test queries (Prompt 18)
  queries = [
    "What time is it right now?",
    "What is 25 * 4 + 10?",
    "Reverse the string 'Hello World'"
  ]

  print("Running example queries:\n")
  for query in queries:
    print(f"📝 Query: {query}\n")
    print("─" * 50)
    try:
      result = agent.invoke({"messages": [{"role": "user", "content": query}]})
      final_response = result["messages"][-1].content
      print(f"✅ Response: {final_response}\n")
    except Exception as e:
      print(f"❌ Error: {str(e)}\n")

  print("🎉 Agent demo complete!")

if __name__ == "__main__":
  main()