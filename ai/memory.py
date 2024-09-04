from abc import ABC, abstractmethod
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from pydantic import BaseModel


class Memory(ABC, BaseModel):
    chat_history: list[BaseMessage] = []
    summary: SystemMessage = SystemMessage(content="")
    max_tokens: int = 8000
    safe_tokens: int = 6000
    summary_model: BaseChatModel

    def _update_summary(self, messages_to_summarize: list[BaseMessage]):
        # Convert messages to a format suitable for summarization
        docs = [Document(page_content=msg.content)
                for msg in messages_to_summarize]

        # Load the summarization chain
        chain = load_summarize_chain(
            self.summary_model, chain_type="map_reduce")

        # Generate the summary
        summary = chain.run(docs)

        # Update the summary
        current_summary = self.summary.content
        self.summary = SystemMessage(
            content=f"{current_summary}\n\nAdditional context: {summary}")

    def _manage_chat_history(self):
        to_summarize = self._trim_chat_history()
        if to_summarize:
            self.update_summary(to_summarize)

    def _trim_chat_history(self):
        to_summarize = []
        total_tokens = sum(len(msg.content.split())
                           for msg in self.chat_history)
        if total_tokens <= self.max_tokens:
            return []
        while total_tokens > self.safe_tokens:
            oldest_message = self.chat_history.pop(0)
            total_tokens -= len(oldest_message.content.split())
            to_summarize.append(oldest_message)
        return to_summarize

    @abstractmethod
    def add_chat_message(self, message: BaseMessage):
        pass

    @abstractmethod
    def add_chat_messages(self, messages: list[BaseMessage]):
        pass


class BasicMemory(Memory):

    def add_chat_message(self, message: BaseMessage):
        self.chat_history.append(message)
        self._manage_chat_history()

    def add_chat_messages(self, messages: list[BaseMessage]):
        self.chat_history.extend(messages)
        self._manage_chat_history()
