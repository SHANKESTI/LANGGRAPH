from langgraph.graph import StateGraph,START,END
from langchain_ollama import ChatOllama
from langchain_core.messages import AnyMessage,AIMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt,Command
from typing import TypedDict,Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
load_dotenv()

model=ChatOllama(
    model='phi3'
)

class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]



def chat_node(state:ChatState):
    decision=interrupt({
        'type':'approval',
        'reason':'Model is about to answer the question.',
        'question':state['messages'][-1].content,
        'instruction':'Approve this question? yes/no'
})
    
    if decision['approved']=='no':
        return {'messages':[AIMessage(content="Not Approved.")]}
    else:
        response=model.invoke(state['messages'])
        return {'messages':[response]}
    


graph=StateGraph(ChatState)
graph.add_node('chat_node',chat_node)
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)
checkpointer=InMemorySaver()
config={"configurable":{"thread_id":"User 1"}}
workflow=graph.compile(checkpointer=checkpointer)

initial_state={
    'messages':[
        ("user","Explain Gradient Descent in very simple terms.")
    ]
}

response=workflow.invoke(initial_state,config=config)

print(response)



