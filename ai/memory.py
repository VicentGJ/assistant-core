import json
import os
from abc import ABC, abstractmethod
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser, BaseMessage, HumanMessage, AIMessage, SystemMessage
from pydantic.v1 import BaseModel, Field


class Memory(ABC, BaseModel):
    chat_history: list[BaseMessage] = []
    summary: SystemMessage = SystemMessage(content="")
    max_tokens: int = 8000
    safe_tokens: int = 6000
    summary_model: BaseChatModel | None = None

    def _update_summary(self, messages_to_summarize: list[BaseMessage]):
        # Convert messages to a format suitable for summarization
        text_to_summarize = "\n\n".join(
            f"<{msg.type}> Message: {msg.content}"
            for msg in messages_to_summarize
        )

       # Perform summarization using custom LCEL chain
        summary = self._summarize_text(text_to_summarize)

        # Update the summary
        current_summary = self.summary.content
        if current_summary == "":
            self.summary.content = f"This is a summary of the older messages in the conversation {
                summary}"
        else:
            final_summary = self._summarize_text(f'''This is a conversation where this is older:\n{
                current_summary}\n\nAnd this is more recent:\n{summary}''')
            self.summary.content = f"This is a summary of the older messages in the conversation {
                final_summary}"

    def _summarize_text(self, text: str) -> str:
        # Prompt for summarization
        summarize_prompt = ChatPromptTemplate.from_template(
            "Summarize the following conversation in a concise manner:\n\n{text}\n\n \
            Your summary must include details about both the User queries and the Ai responses. Summary:"
        )

        # LCEL chain for summarization
        summarize_chain = (
            summarize_prompt
            | self.summary_model
            | StrOutputParser()
        )

        # Run the chain
        return summarize_chain.invoke({"text": text})

    def _manage_chat_history(self):
        to_summarize = self._trim_chat_history()
        if to_summarize:
            self._update_summary(to_summarize)

    def _trim_chat_history(self):
        to_summarize = []
        total_tokens = sum(len(msg.content.split())
                           for msg in self.chat_history)
        if total_tokens < self.max_tokens:
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


class FileMemory(Memory):
    path: str = Field(...,
                      description="Path to the file where memory will be stored")

    def __init__(self, **data):
        super().__init__(**data)
        self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                data = json.load(f)
                self.chat_history = self._deserialize_messages(
                    data['chat_history'])
                self.summary = SystemMessage(content=data['summary'])

    def _save_memory(self):
        data = {
            'chat_history': self._serialize_messages(self.chat_history),
            'summary': self.summary.content
        }
        with open(self.path, 'w') as f:
            json.dump(data, f)

    def _serialize_messages(self, messages: list[BaseMessage]) -> list[dict]:
        return [{'type': msg.__class__.__name__, 'content': msg.content} for msg in messages]

    def _deserialize_messages(self, data: list[dict]) -> list[BaseMessage]:
        message_types = {
            'HumanMessage': HumanMessage,
            'AIMessage': AIMessage,
            'SystemMessage': SystemMessage
        }
        return [message_types[msg['type']](content=msg['content']) for msg in data]

    def add_chat_message(self, message: BaseMessage):
        super().add_chat_message(message)
        self._save_memory()

    def add_chat_messages(self, messages: list[BaseMessage]):
        super().add_chat_messages(messages)
        self._save_memory()

    def _update_summary(self, messages_to_summarize: list[BaseMessage]):
        super()._update_summary(messages_to_summarize)
        self._save_memory()
