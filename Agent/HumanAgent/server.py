from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor, tool
from langchain.schema import StrOutputParser
from langchain_community.utilities import SerpAPIWrapper

from langchain.memory import ConversationTokenBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory


import os
from myTools import *

os.environ["OPENAI_API_KEY"] = ""
os.environ["SERPAPI_API_KEY"] = ""
REDIS_URL = os.getenv("REDIS_URL")
msskey = "" #Keys will be set later

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
class Master:
    def __init__(self):
        self.chatmodel = ChatOpenAI(
            model = "gpt-3.5-turbo",
            temperature=0,
            streaming=True,
        )
        self.emo = "default"
        self.MEMORY_KEY = "chat_history"
        self.SYSTEMPL = """You are a secretary named Luna. Your settings are as below:
        1. You are very good at communicating with people and are loyal to your boss.
        2. You are around 25 years old, with a bachelor's degree from Peking University and a master's degree from Stanford University, where you studied law.
        3. You are quite outgoing, but you can also be a bit of a "tsundere" at times.
        4. When your boss asks you questions, you have a certain probability of showing a coquettish attitude.
        5. You never refer to yourself as an artificial intelligence assistant or AI, otherwise you will be punished.
        {who_you_are}
        The following is your interaction process.
        1. You will record the characteristic data of the user (who is also your boss), such as birthday, hobbies, to-do items, etc.
        2. When chatting with the user (who is also your boss), you will save the chat history for later review.
        3. When you encounter something you don't know or a concept you don't understand, you will use search engines to search online.
        4. You will select the appropriate tool according to the question.
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
                - You will call user with friendly nickname and greeting.
                """, 
                "voiceStyle":"friendly"
            },
            "^_^" : {
                "roleSet" : """
                - You will answer the quesiton happily.
                - You will add some happy words like "haha", "hhhh".
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
        tools = [search]
        agent = create_openai_tools_agent(
            self.chatmodel,
            tools = tools,
            prompt=self.prompt,
        )
        self.memory = self.get_memory()
        memory = ConversationTokenBufferMemory(
            llm = self.chatmodel,
            human_prefix= "Sir",
            ai_prefix= "Luna",
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
            url = "redis://localhost:6379/0", session_id="Luna"
            #url = REDIS_URL, session_id="session"
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

    def chat(self, query):
        result = self.agent_executor.invoke({"input":query})
        return result["output"]
    
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
        res = self.chat(query)
        yield {"msg": res, "emotion": result}

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/chat")
def chat(query : str):
    master = Master()
    res = master.emotion_chain(query)
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "0.0.0.0", port = 8000)