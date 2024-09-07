import yaml
import datetime
import os
import traceback
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chat_models.base import BaseChatModel
from langchain_mistralai import ChatMistralAI
from ai.assistant import Assistant
from ai.memory import BasicMemory, FileMemory
from ai.tools.email import EmailToolkit
from ai.knowledge import KnowledgeSearchTool, get_faiss
from ai.tools.image import ImageGenerationTool
from utils.cli import cli_app
from utils.system_prompts import assistant_description_without_tool_descriptions


def test_assistant_basic(assistant: Assistant):
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


def run_test(assistant: Assistant, test: str):
    with open("testing/test_prompts.yaml", "r") as f:
        try:
            tests = yaml.safe_load(f)["tests"][test]
        except KeyError:
            print(f"Test {test} not found in testing/test_prompts.yaml")
            return

    for test_name, test_data in tests.items():
        # Magenta color for test names
        print(f"\n\033[95mRunning test: {test_name}\033[0m\n")
        prompts = test_data["prompts"]

        for i, prompt in enumerate(prompts):
            # Green color for steps
            print(f"\033[92mStep {i+1}: {prompt}\033[0m")
            print("\n" + "="*50 + "\n")  # Add a separator line

            try:
                response = assistant.get_response(prompt)
                print("\033[94m" + "-"*50)  # Start blue formatting
                print("Assistant:")
                print(response.content)
                print("-"*50 + "\033[0m")  # End blue formatting
            except Exception as e:
                # Red color for errors
                print(
                    "\033[91mError: An issue occurred while processing your request.")
                print(f"Details: {str(e)}")
                print("Stack trace:")
                traceback.print_exc()
                print("The application will now exit.\033[0m")
                break  # Exit the loop on error

            print("\n" + "="*50 + "\n")  # Add a separator line

        # Yellow color for final summary
        print("\n\033[93mFinal Summary:\033[0m")
        print(assistant.memory.summary.content)
        # Add a magenta separator line
        print("\n\033[95m" + "="*50 + "\033[0m\n")
