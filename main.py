
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from tavily import TavilyClient

tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    :param query:
    The query to search for
    :return:
    The search result
    """

    print(f"Searching for {query}")
    return tavily.search(query=query)

llm = ChatOllama(model="qwen2.5:7b", temperature=0)
tools = [search]
agent = create_agent(model=llm, tools=tools)



def main():
    result = agent.invoke({"messages": HumanMessage(content="Search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")})
    print(result)

if __name__ == "__main__":
    main()