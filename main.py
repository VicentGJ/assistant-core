import datetime
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from ai.assistant import Assistant
from ai.memory import BasicMemory, FileMemory
from ai.tools.email import EmailToolkit
from ai.knowledge import KnowledgeSearchTool, get_faiss
from ai.tools.image import ImageGenerationTool
from testing.test import test_assistant_single_tool
from utils.cli import cli_app
from utils.system_prompts import assistant_description_with_tool_descriptions
from dotenv import load_dotenv

load_dotenv()


def main():
    # Setup tools
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
        knowledge_tool,
        ImageGenerationTool()
    ] + email_toolkit.get_tools()

    # Setup model
    model = ChatMistralAI(model_name="open-mistral-nemo")

    # # Setup file memory
    # current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # file_memory = FileMemory(
    #     path="memory_files/" + current_time + ".json",
    #     summary_model=model,
    #     max_tokens=200,
    #     safe_tokens=150
    # )

    # # Setup assistant
    # assistant = Assistant(
    #     model=model,
    #     tools=tools,
    #     memory=file_memory,
    #     description=assistant_description_with_tool_descriptions
    # )

    # cli_app(assistant)
    test_assistant_single_tool(model)


if __name__ == "__main__":
    main()
