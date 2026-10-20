from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain.agents.structured_output import ToolStrategy


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The url of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="The list of sources used to generate the answer")


llm = ChatOllama(model="llama3.2:3b", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=ToolStrategy(AgentResponse),)


def main():
    result = agent.invoke({"messages": HumanMessage(
        content="Search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details and links")})
    print(result)


if __name__ == "__main__":
    main()
