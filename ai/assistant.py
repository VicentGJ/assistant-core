from langchain.chat_models.base import BaseChatModel
from langchain.tools import BaseTool
from langchain.schema import HumanMessage, BaseMessage
from langgraph.prebuilt import create_react_agent
from pydantic.v1 import BaseModel, Field
from .memory import Memory, BasicMemory


class ResponseSchema(BaseModel):
    content: str = Field(..., description="The content of the response")
    tool_call: dict | None = Field(
        None, description="Optional tool call information")


class Assistant:
    """
    The Assistant class provides an interface for interacting with a conversational AI agent.
    It manages the conversation history, processes user inputs, generates responses using a
    ReAct agent, and handles tool calls if necessary.

    Attributes:
        graph (ReActAgent): The ReAct agent used to generate responses.
        memory (Memory): The memory object used to store the conversation history and summaries.

    Methods:
        __init__(model: BaseChatModel, tools: list[BaseTool], memory: Memory | None, description: str | None) -> None:
            Initializes the Assistant with the given model, tools, memory, and description.

        _create_input_messages() -> list[BaseMessage]:
            Creates a list of input messages for the ReAct agent, including the summary and chat history.

        get_response(input: str) -> ResponseSchema:
            Processes the user's input, generates a response using the ReAct agent, and handles
            potential tool calls. Updates the assistant's memory with the conversation history.

        print_response(input: str) -> None:
            Processes the user's input, generates a response using the ReAct agent, and prints
            the response. Updates the assistant's memory with the conversation history.

    Example usage:
        model = SomeChatModel()
        tools = [SomeTool()]
        memory = SomeMemory()
        description = "Some description"
        assistant = Assistant(model, tools, memory, description)

        response = assistant.get_response("What is the weather like today?")
        print(response.content)
    """

    def __init__(
            self,
            model: BaseChatModel,
            tools: list[BaseTool],
            memory: Memory | None,
            description: str | None):

        self.graph = create_react_agent(
            model=model, tools=tools, state_modifier=description
        )
        self.memory = memory or BasicMemory()

    def _create_input_messages(self) -> list[BaseMessage]:
        messages = []
        if self.memory.summary.content:
            messages.append(self.memory.summary)
        messages.extend(self.memory.chat_history)
        return messages

    def get_response(self, input: str) -> ResponseSchema:
        """
        Get a response from the assistant based on the input provided.

        This method processes the input, generates a response using the ReAct agent,
        and handles potential tool calls. It also updates the assistant's memory
        with the conversation history.

        Args:
            input (str): The user's input message.

        Returns:
            ResponseSchema: A schema containing the response content and optional tool call information.

        The method performs the following steps:
        1. Adds the user's input to the chat history.
        2. Prepares the input messages for the ReAct agent.
        3. Streams the response from the agent.
        4. Processes the final result to extract the response and any tool calls.
        5. Updates the memory with the assistant's response.
        6. Returns a ResponseSchema object with the appropriate content and tool call information.

        If no response is generated, it returns a ResponseSchema with a default message.
        """
        human_message = HumanMessage(content=input)
        self.memory.add_chat_message(human_message)

        inputs = {"messages": self._create_input_messages()}

        final_result = None
        for chunk in self.graph.stream(inputs, stream_mode="values"):
            final_result = chunk

        if final_result:
            # Search for tool call message and final response
            tool_call_message = None
            final_response = None
            for message in reversed(final_result["messages"]):
                if message.type == "ai":
                    if not final_response:
                        final_response = message
                    if message.content == '' and message.additional_kwargs.get("tool_calls"):
                        tool_call_message = message
                        break

            # Add final response to memory
            if final_response:
                self.memory.add_chat_message(final_response)

            # Return appropriate ResponseSchema
            if tool_call_message:
                tool_call = tool_call_message.additional_kwargs["tool_calls"][0]
                return ResponseSchema(content=final_response.content, tool_call=tool_call)
            elif final_response:
                return ResponseSchema(content=final_response.content)

        return ResponseSchema(content="No response generated")

    def print_response(self, input: str):
        human_message = HumanMessage(content=input)
        self.memory.add_chat_message(human_message)

        inputs = {"messages": self._create_input_messages()}

        for chunk in self.graph.stream(inputs, stream_mode="values"):
            message = chunk["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()

        # Assuming the last message in the stream is the AI's response
        ai_message = chunk["messages"][-1]
        self.memory.add_chat_message(ai_message)
