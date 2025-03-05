
from langchain.agents import create_openai_tools_agent, AgentExecutor, tool

from langchain_community.utilities import SerpAPIWrapper
from langchain_openai import OpenAIEmbeddings

@tool
def search(query : str):
    """Only when you need to know the inforamtion realtime or the thing you do not know"""
    serp = SerpAPIWrapper()
    result = serp.run(query)
    print("real time search result", result)
    return result
