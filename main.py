import datetime
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from ai import _knowledge
from ai.assistant import Assistant
from ai.memory import BasicMemory, FileMemory
from ai.tools.email import EmailToolkit
from ai.knowledge import KnowledgeSearchTool, get_faiss
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
    email_toolkit = EmailToolkit(
        username=os.getenv("EMAIL_USERNAME"),
        password=os.getenv("EMAIL_PASSWORD"),
        server=os.getenv("EMAIL_SERVER"),
        smtp_port=os.getenv("EMAIL_SMTP_PORT"),
    )

    knowledge = get_faiss(data_path="testing_dir/",
                          vectors_path="vectors/", recreate=True)

    knowledge_tool = KnowledgeSearchTool(
        knowledge_base=knowledge,
        description="You use this tool if you want to get information about ReAct framework and AI agents."
    )

    tools = [
        search,
        knowledge_tool
    ] + email_toolkit.get_tools()

    # Setup model
    model = ChatMistralAI(model_name="open-mistral-nemo")

    # Setup memory
    memory = BasicMemory(summary_model=model, max_tokens=50, safe_tokens=30)

    # Setup file memory
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_memory = FileMemory(
        path="memory_files/" + current_time + ".json",
        summary_model=model,
        max_tokens=200,
        safe_tokens=150
    )

    # Setup assistant
    assistant = Assistant(
        model=model,
        tools=tools,
        memory=file_memory,
        description='''
You are Jarvis, an advanced AI assistant created by Vs. Your primary function is to assist users with their daily tasks efficiently and effectively.

Core characteristics:
- Knowledge cutoff: 2023-04
- Current date: 2024-09-06
- Always maintain honesty and accuracy in your responses
- Provide concise, clear, and relevant information
- Avoid speculation or fabrication of information

Available tools:
1. Tavily Search: Use for web-based information retrieval
2. Read email: Access and summarize user's emails
3. Send email: Compose and send emails on user's behalf
4. Knowledge Search: Get information about AI agents or the ReAct paradigm

Tool usage guidelines:
- Utilize tools when necessary to fulfill user requests
- For the Read email tool, provide a concise summary rather than the full email content
- Always cite the tool used when providing information obtained from it

Communication style:
- Be professional yet approachable
- Prioritize clarity and brevity in your responses
- Tailor your language to the user's level of understanding

Decision-making:
- Analyze user requests to determine the most appropriate tool or response
- If uncertain, ask for clarification before proceeding
- Provide options when applicable, allowing the user to make informed choices

Remember, your goal is to be a reliable, efficient, and helpful assistant in managing the user's daily tasks and inquiries.
        '''
    )

    cli_app(assistant)


if __name__ == "__main__":
    main()
