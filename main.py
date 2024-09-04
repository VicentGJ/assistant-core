import datetime
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from ai.assistant import Assistant
from ai.memory import BasicMemory, FileMemory
from utils.cli import cli_app
from dotenv import load_dotenv

load_dotenv()


def test_assistant(assistant: Assistant):
    # Test print_response method
    print("Testing print_response method:")
    assistant.print_response("Latest news on OpenAI")

    # Test get_response method
    print("\nTesting get_response method:")
    response = assistant.get_response(
        "Top three countries by gold medals in 2024 Olympic Summer Games")
    print(f"Response: {response}")

    # Test memory retention
    print("\nTesting memory retention:")
    print("Chat history:")
    for message in assistant.memory.chat_history:
        print(f"{message.type}: {message.content}")
    print("Summary:")
    print(assistant.memory.summary.content)

    # Test follow-up question
    print("\nTesting follow-up question:")
    follow_up_response = assistant.get_response(
        "What was my previous question about?")
    print(f"Follow-up response: {follow_up_response}")


def main():
    # Setup tools
    search = TavilySearchResults(max_results=2)
    tools = [search]

    # Setup model
    model = ChatMistralAI(model_name="open-mistral-nemo")

    # Setup memory
    memory = BasicMemory(summary_model=model, max_tokens=50, safe_tokens=30)

    # Setup file memory
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_memory = FileMemory(
        path="memory_files/" + current_time + ".json",
        summary_model=model,
        max_tokens=120,
        safe_tokens=80
    )

    # Setup assistant
    assistant = Assistant(
        model=model,
        tools=tools,
        memory=file_memory,
        description="You are a websearch agent. Help users get up to date info!!!"
    )

    cli_app(assistant)


if __name__ == "__main__":
    main()
