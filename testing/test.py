import yaml
import datetime
import os
import traceback
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chat_models.base import BaseChatModel
from langchain_mistralai import ChatMistralAI
from ai import assistant
from ai.assistant import Assistant
from ai.memory import BasicMemory, FileMemory
from ai.tools.email import EmailToolkit
from ai.knowledge import KnowledgeSearchTool, get_faiss
from ai.tools.image import ImageGenerationTool
from utils.cli import cli_app
from utils.system_prompts import assistant_description_without_tool_descriptions, assistant_without_tools


def setup_tools():
    email_toolkit = EmailToolkit(
        username=os.getenv("EMAIL_USERNAME"),
        password=os.getenv("EMAIL_PASSWORD"),
        server=os.getenv("EMAIL_SERVER"),
        smtp_port=os.getenv("EMAIL_SMTP_PORT"),
    )

    knowledge = get_faiss(data_path="testing_dir/",
                          vectors_path="vectors/", recreate=False)

    knowledge_tool = KnowledgeSearchTool(
        knowledge_base=knowledge,
        description="You use this tool if you want to get information about ReAct framework and AI agents."
    )

    tools = [
        TavilySearchResults(max_results=4),
        # knowledge_tool,
        ImageGenerationTool()
    ] + email_toolkit.get_tools()

    return tools


def test_assistant_conversational(model: BaseChatModel):
    assistant = Assistant(
        model=model,
        memory=BasicMemory(),
        description=assistant_without_tools
    )
    run_test(assistant, "conversational")


def test_assistant_single_tool(model: BaseChatModel, name: str = "Nemo"):
    assistant = Assistant(
        model=model,
        memory=BasicMemory(),
        tools=setup_tools(),
        name=name,
        description=assistant_description_without_tool_descriptions
    )
    run_test(assistant, "single_function_call")


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

        chat_history_yaml = {
            "chat_history": [
                {"type": message.type, "content": message.content}
                for message in assistant.memory.chat_history
            ],
            "summary": assistant.memory.summary.content
        }

        # Serialize the final chat history to a YAML file with datetime
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        chat_history_yaml = {
            "chat_history": [
                {"type": message.type, "content": message.content}
                for message in assistant.memory.chat_history
            ],
            "summary": assistant.memory.summary.content
        }

        with open(f"testing/results/test_{assistant.name}_{test_name}_{current_time}.yaml", "w") as f:
            yaml.dump(chat_history_yaml, f)

        print(f"\n\033[93mFinal Chat History and Summary saved to "
              f"testing/results/chat_{assistant.name}_{test_name}_{current_time}.yaml\033[0m")

        # Add a magenta separator line
        print("\n\033[95m" + "="*50 + "\033[0m\n")
