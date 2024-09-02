from abc import ABC, abstractmethod
from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel


class Memory(ABC, BaseModel):
    @abstractmethod
    def add_chat_message(self, message: BaseMessage):
        pass

    @abstractmethod
    def add_chat_messages(self, messages: list[BaseMessage]):
        pass


class BasicMemory(Memory):
    chat_history: list[BaseMessage] = []
    summary: SystemMessage = SystemMessage(content="")

    def add_chat_message(self, message: BaseMessage):
        self.chat_history.append(message)
        # TODO: Check if message stays within safe bounds. Update Summary if not

    def add_chat_messages(self, messages: list[BaseMessage]):
        self.chat_history.extend(messages)
        # TODO: Check if messages stay within safe bounds. Update Summary if not
