from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from pydantic import BaseModel,Field
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage,HumanMessage
from langgraph.graph.message import add_messages

model=ChatOllama(
    model='phi3'
)

class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

def chat_node(state:ChatState):
    messages=state['messages']
    content=model.invoke(messages)

    return {"messages":content}

graph=StateGraph(ChatState)
graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

workflow=graph.compile()

initial_state={
    'messages':[HumanMessage(content="Write a summary on the Indian Cricket")]
}


response=workflow.invoke(initial_state)

print(response)