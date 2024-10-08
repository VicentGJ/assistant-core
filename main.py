import datetime
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from assistant_core.assistant import Assistant
from assistant_core.memory import BasicMemory, FileMemory
from assistant_core.tools.email import EmailToolkit
from assistant_core.knowledge import KnowledgeSearchTool, get_faiss
from assistant_core.tools.image import ImageGenerationTool
from testing.test import test_assistant_multiple_tools, test_assistant_single_tool
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

    # knowledge = get_faiss(data_path="testing_dir/",
    #                       vectors_path="vectors/", recreate=False)

    # knowledge_tool = KnowledgeSearchTool(
    #     knowledge_base=knowledge,
    #     description="You use this tool if you want to get information about ReAct framework and AI agents."
    # )

    tools = [
        # TavilySearchResults(max_results=4),
        # knowledge_tool,
        ImageGenerationTool(),
    ]  # + email_toolkit.get_tools()

    # Setup model
    mistral = ChatMistralAI(model="open-mistral-nemo")
    gateway = ChatOpenAI(
        model="spark",
        api_key=os.getenv("APIGATEWAY_KEY"),
        base_url="https://apigateway.avangenio.net",
    )
    huggingface_enpodint = HuggingFaceEndpoint(
        repo_id="CohereForAI/c4ai-command-r-plus-08-2024",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )
    huggingface = ChatHuggingFace(llm=huggingface_enpodint)
    openai = ChatOpenAI(model="gpt-4o-2024-08-06")
    ollama = ChatOllama(model="mistral-nemo", num_predict=1024)

    # # Setup file memory
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_memory = FileMemory(
        path="memory_files/" + current_time + ".json",
        summary_model=mistral,
        max_tokens=200,
        safe_tokens=150,
    )

    # Setup assistant
    assistant = Assistant(
        model=huggingface,
        tools=tools,
        memory=file_memory,
        description=assistant_description_with_tool_descriptions,
    )

    cli_app(assistant)

    # test_assistant_single_tool(mistral, name="nemo")
    # test_assistant_single_tool(gateway, name="gateway")
    # test_assistant_single_tool(openai, name="gpt4o")

    # test_assistant_single_tool(mistral, name="nemo")
    # test_assistant_multiple_tools(openai, name="gpt4o")
    # test_assistant_single_tool(ollama, name="ollama")


if __name__ == "__main__":
    main()
