import os
from datetime import datetime

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
        allowed_names = {"__builtins__": None}
        result = eval(expression, allowed_names, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    load_dotenv()

    print("🚀 Starting application...")

    # Optional GitHub token (used later for GitHub-hosted models)
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token:
        print("GitHub token loaded successfully!")
    else:
        print("GitHub token not found (OK for now).")

    # Required OpenAI key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("OPENAI_API_KEY not found!")
        return

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=openai_key,
    )

    print("ChatOpenAI model initialized successfully!")

    # Tool list (Prompt 10)
    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description=(
                "Use this tool to perform mathematical calculations. "
                "Provide a full math expression as input, such as '25 * 4 + 10'. "
                "This tool should be used whenever a calculation is required."
            ),
        )
    ]

    # Test query (still no agent yet)
    query = "What is 25 * 4 + 10?"
    response = llm.invoke([HumanMessage(content=query)])
    print("Response:", response.content)


if __name__ == "__main__":
    main()
