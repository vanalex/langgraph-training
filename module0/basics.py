from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_tavily import TavilySearch

load_dotenv()

tavily_search = TavilySearch(max_results=3)

@dataclass
class ChatConfig:
    model: str
    temperature: float
    chat: ChatOpenAI = field(init=False)

    def __post_init__(self):
        self.chat = ChatOpenAI(model=self.model, temperature=self.temperature)

    def invoke_chat(self, messages: list[str | BaseMessage]) -> AIMessage:
        return self.chat.invoke(messages)

    def chat_research(self, query: str = "What is LangGraph?") -> Any:
        data = tavily_search.invoke({"query": query})
        search_docs = data.get("results", data)
        return search_docs

def main():
    print("Hello world")

    # Test with GPT-3.5
    chat_config = ChatConfig(model="gpt-3.5-turbo-0125", temperature=0)
    chat_result = chat_config.invoke_chat(["Hello world"])
    print(chat_result)

    search_docs = chat_config.chat_research()
    print(search_docs)

    # Test with GPT-4
    chat_config = ChatConfig(model="gpt-4o", temperature=0)
    chat_result = chat_config.invoke_chat(["Hello world"])
    print(chat_result)

    search_docs = chat_config.chat_research()
    print(search_docs)

if __name__ == "__main__":
    main()