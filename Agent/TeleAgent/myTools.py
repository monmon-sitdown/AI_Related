
from langchain.agents import create_openai_tools_agent, AgentExecutor, tool

from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings

@tool
def search(query : str):
    """Only when you need to know the inforamtion realtime or the thing you do not know"""
    serp = SerpAPIWrapper()
    result = serp.run(query)
    print("real time search result", result)
    return result

@tool
def get_info_from_local_db(query : str):
    """Only when you need to answer the question about MBTI"""
    client = Qdrant(
        QdrantClient(path = "./local_qdrand"),
        "local_documents",
        OpenAIEmbeddings(),
    )
    retriever = client.as_retriever(search_type = "mmr")
    result = retriever.get_relevant_documents(query)
    return result