from langgraph.graph import StateGraph,START,END
from langchain_ollama import ChatOllama
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

model=ChatOllama(
    model="phi3"
)

class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

def chat_node(state:ChatState):
    message=state['messages']
    response=model.invoke(message).content
    return {
        "messages":[AIMessage(content=response)]
    }

conn=sqlite3.connect(database='chatbot.db',check_same_thread=False)

checkpointer=SqliteSaver(conn=conn)

graph=StateGraph(ChatState)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)


chatbot=graph.compile(checkpointer=checkpointer)

config={"configurable":{"thread_id":"user1"}}

response=chatbot.invoke({
    "messages":[HumanMessage(content="Hello Buddy")]},
    config=config
)

print(response)