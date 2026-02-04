import os 
from datetime import datetime
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool


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

def get_current_time(_: str) -> str:
    """
    Returns the current date and time as a formatted string.
    Uses datetime.now().strftime("%Y-%m-%d %H:%M:%S").
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

 
def main():
    load_dotenv()
    print("🚀 Starting application...")

    # Optional GitHub token (used later for GitHub-hosted models)
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("❌ GITHUB_TOKEN not found in environment variables.")
        print("👉 Please set your GitHub token in a .env file or as an environment variable.")
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

    # Tool list (Prompt 10)
    # Commenting out the Calculator tool for Prompt 12
    # tools = [
    #     Tool(
    #         name="Calculator",
    #         func=calculator,
    #         description=(
    #             "Use this tool to perform mathematical calculations. "
    #             "Provide a full math expression as input, such as '25 * 4 + 10'. "
    #             "This tool should be used whenever a calculation is required."
    #         ),
    #     )
    # ]

    # Test query (Prompt 12: Without tool)
    query = "What time is it right now?"
    try:
        response = llm.invoke([HumanMessage(content=query)])
        print("Response:", response.content)
    except Exception as e:
        print("❌ Error during model invocation:", str(e))
        print("👉 Please check that your GITHUB_TOKEN is valid and has access to https://models.github.ai/inference.")
        print("   If you are using a personal access token, ensure it has the correct scopes.")
        return

if __name__ == "__main__":
    main()

