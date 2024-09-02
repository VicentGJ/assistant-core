from langchain.chat_models.base import BaseChatModel
from langchain.tools import BaseTool
from langchain.schema import HumanMessage
from langgraph.prebuilt import create_react_agent
from .memory import Memory, BasicMemory


class Assistant:

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

    def get_response(self, input: str):
        human_message = HumanMessage(content=input)
        self.memory.add_chat_message(human_message)

        inputs = {"messages": self.memory.chat_history + [human_message]}

        final_result = None
        for chunk in self.graph.stream(inputs, stream_mode="values"):
            final_result = chunk

        if final_result:
            ai_message = final_result["messages"][-1]
            self.memory.add_chat_message(ai_message)
            return ai_message.content
        return None

    def print_response(self, input: str):
        human_message = HumanMessage(content=input)
        self.memory.add_chat_message(human_message)

        inputs = {"messages": self.memory.chat_history + [human_message]}

        for chunk in self.graph.stream(inputs, stream_mode="values"):
            message = chunk["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()

        # Assuming the last message in the stream is the AI's response
        ai_message = chunk["messages"][-1]
        self.memory.add_chat_message(ai_message)
