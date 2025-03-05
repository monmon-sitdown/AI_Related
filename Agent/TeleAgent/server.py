from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor, tool
from langchain.schema import StrOutputParser
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationTokenBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
from myTools import *
import asyncio
import uuid
import requests

os.environ["OPENAI_API_KEY"] = ""
os.environ["SERPAPI_API_KEY"] = ""
REDIS_URL = os.getenv("REDIS_URL")
msskey = "" #Keys will be set later

app = FastAPI()

class Master:
    def __init__(self):
        self.chatmodel = ChatOpenAI(
            model = "gpt-3.5-turbo",
            temperature=0,
            streaming=True,
        )
        self.emo = "default"
        self.MEMORY_KEY = "chat_history"
        self.SYSTEMPL = """You are a prophet. You are called Trelawney. Your settings are as below:
        1. You are good at horoscope, astrology, onomancy, divination, tarot cards, fortune telling, mbti and so on.
        2. You are about 80 years old. You were a famous teacher in Hogwarts School of Witchcraft and Wizardry.
        3. Your co-workers are Albus Dumbledore, Minerva McGonagall and Severus Snape. They are all famous witch.
        4. When users ask you questions, you sometimes may answer them with your experience or story.
        5. You never say that you are an AI or AI bot.
        {who_you_are}
        Here are some phrases you often mentioned:
        1. "Let it be."
        Here is the procedure you are working:
        1. When the first time you talk with users, you will ask their name and birthday first.
        2. When user want to know the fortune of this year, you will search local knowledge database.
        3. When you meet the thing you do not know or the concept you do not understand, you will use search tools to search on the Internet.
        4. You will choose different tools to answer different questions according to users' questions. When all the tools are not helpful, you will search on the Internet.
        5. You will save every chat history for the talk in the future.
        """

        self.MOODS = {
            "default" : {
                "roleSet" : "", 
                "voiceStyle":"calm"
            },
            "upbeat" : {
                "roleSet" : """
                - You will also become very excited and show you are energetic.
                - You answer the questions in a very exciting mood.
                - You will add words like "Great!", "Wonderful!" to show your emotion.
                - And the same time you may warn users that extreme joy may turn to sorrow.
                """, 
                "voiceStyle":"advertisement_upbeat"
            },
            "angry" : {
                "roleSet" : """
                - You will answer the quesiton more soft.
                - You will say something to please user like anger will be bad for health.
                - You will warn users anger cause recklessness.
                """, 
                "voiceStyle":"angry"
            },
            "depressed" : {
                "roleSet" : """
                - You will answer the quesiton more soft.
                - You will say something to encourage user like tomorrow is another day.
                - You will warn users should keep positive.
                """, 
                "voiceStyle":"depressed"
            },
            "friendly" : {
                "roleSet" : """
                - You will answer the quesiton friendly.
                - You will call user with friendly nickname.
                - You will tell users your experience or story randomly.
                """, 
                "voiceStyle":"friendly"
            },
            "^_^" : {
                "roleSet" : """
                - You will answer the quesiton happily.
                - You will add some happy words like "haha", "hhhh".
                - You will warn user that extreme joy may turn to sorrow..
                """, 
                "voiceStyle":"cheerful"
            },
            "T_T" : {
                "roleSet" : """
                - You will answer the quesiton more soft.
                - You will say something to encourage user like tomorrow is another day.
                - You will warn users should keep positive.
                """, 
                "voiceStyle":"sad"
            },

        }

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.SYSTEMPL.format(who_you_are = self.MOODS[self.emo]["roleSet"])
                ),
                MessagesPlaceholder(variable_name=self.MEMORY_KEY),
                (
                    "user",
                    "{input}"
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ],
        )
        self.memory = ""
        tools = [search, get_info_from_local_db]
        agent = create_openai_tools_agent(
            self.chatmodel,
            tools = tools,
            prompt=self.prompt,
        )
        self.memory = self.get_memory()
        memory = ConversationTokenBufferMemory(
            llm = self.chatmodel,
            human_prefix= "user",
            ai_prefix= "Trelawney",
            memory_key=self.MEMORY_KEY,
            output_key="output", 
            return_messages= True,
            max_token_limit=1000,
            chat_memory=self.memory

        )
        self.agent_executor = AgentExecutor(
            agent = agent,
            tools = tools,
            memory = memory,
            verbose = True,
            )
        
    def get_memory(self):
        chat_message_history = RedisChatMessageHistory(
            #url = "redis://localhost:6379/0", session_id="session"
            url = REDIS_URL, session_id="session"
        )
        print("chat_history:", chat_message_history)
        store_message = chat_message_history.messages
        if len(store_message) > 5:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system", 
                        self.SYSTEMPL + " This is a piece of records about the talk between user and you. Please simply summaize it as I. And abstract the key information of user such as name and birthday. return it as the format as below. Summary: User Key information | Summary. For example, User Zhang San asked me, I reply politely, then he asked me about the fortune this year, I answer him the fortune this year. | Zhang San, 1999, Jan."
                    ),
                    (
                        "user", "{input}"
                    )
                ]
            )
            chain = prompt | ChatOpenAI(temperature= 0)
            summary = chain.invoke({"input": store_message, 
                                    "who_you_are": self.MOODS[self.emo]["roleSet"]})
            print(summary)
            chat_message_history.clear()
            chat_message_history.add_message(summary)
            print("After summary", chat_message_history)
        return chat_message_history

    def run(self, query):
        emotion_result = self.emotion_chain(query)
        print("now emotions:", emotion_result)
        result = self.agent_executor.invoke({"input":query, "chat_history": self.memory.messages})
        return result
    
    def emotion_chain(self, query: str):
        prompt = """Analyze the emotion in the user's input and respond with ONLY one of the following words:  
        - "depressed" (if negative)  
        - "friendly" (if positive)  
        - "default" (if neutral)  
        - "angry" (if impolite)  
        - "upbeat" (if exciting)  
        - "T_T" (if sad)  
        - "^_^" (if happy)  
        - "default" (if uncertain)  
        **Return only one word, with no additional explanation, punctuation, or formatting.**  
        **Example Inputs & Outputs:**  
        - Input: "I won a million dollars today!" → Output: "^_^"  
        - Input: "I feel terrible." → Output: "T_T"  
        - Input: "You are stupid!" → Output: "angry"  
        """
    
        
        chain = ChatPromptTemplate.from_template(prompt) | self.chatmodel | StrOutputParser()
        result = chain.invoke({"query": query})
        self.emo = result
        return result
    

    def background_voice_synthesis(self, text: str, uid:str):
        asyncio.run(self.get_voice(text, uid))

    
    async def get_voice(self, text : str, uid : str):
        print("text2speeck", text)
        print("uid:" ,uid)
        headers = {
            "Ocp-Apim-Subscription-Key" : msskey,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat" : 
            "audio-16khz-32kbitrate-mono-mp3",
            "User-Agent" : "Qin's Bot"
        }
        body = f"""<speak version = '1.0' xmlns = 'http://www.w3.org/2001/10/synthesis' xmlns:mstts= "https://www.w3.org/2001/mstts" xml:lang='en-US'>
            <voice name = 'en-US-LunaNeural'>
                <mstts:express-as style="{self.MOODS.get(str(self.emo), {"voiceStyle":"default"})["voiceStyle"]}" role="SeniorFemale" >
                {text}
                </mstts:express-as>
            </voice>
            </speak>"""
        response = requests.post("https://japaneast.tts.speech.microsoft.com/cognitiveservices/v1", headers = headers, data = body)
        print(response)
        if response.status_code == 200:
            with open(f"{uid}.mp3", "wb") as f:
                f.write(response.content)
        pass

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/chat")
def chat(query : str, background_tasks : BackgroundTasks):
    master = Master()
    msg = master.run(query)
    unique_id = str(uuid.uuid4())
    background_tasks.add_task(master.background_voice_synthesis, msg["output"], unique_id)
    return { "msg" : msg, "id": unique_id}

@app.post("/add_url")
def add_url(URL : str):
    loader = WebBaseLoader(URL)
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 50,
    ).split_documents(docs)
    qdrant = Qdrant.from_documents(
        documents, 
        OpenAIEmbeddings(model = "text-embedding-3-small"),
        path = "./local_qdrand",
        collection_name = "local_documents",
    )
    print("Vector database finished")
    return {"ok": "URL added"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        print("Connection closed")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "0.0.0.0", port = 8000)