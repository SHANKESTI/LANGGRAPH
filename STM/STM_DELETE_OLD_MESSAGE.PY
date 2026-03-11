from langgraph.graph import StateGraph,START,END,MessagesState
from langchain_ollama import ChatOllama
from langchain.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver


model=ChatOllama(
    model='phi3'
)

def chat_node(state:MessagesState):
    response=model.invoke(state['messages'])
    return {"messages":response}


def delete_old(state:MessagesState):
    msg=state['messages']
    if len(msg)>10:
        to_remove=msg[:6]
        return {"messages":[RemoveMessage(id=m.id)for m in to_remove]}
    
    return {}


graph=StateGraph(MessagesState)
graph.add_node('chat_node',chat_node)
graph.add_node('delete_old',delete_old)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node','delete_old')
graph.add_edge('delete_old',END)

checkpointer=InMemorySaver()

workflow=graph.compile(checkpointer=checkpointer)

config={"configurable":{"thread_id":"t1"}}

print(workflow.invoke({"messages": [{"role": "user", "content": "Hi, I'm Nitish"}]}, config))
print(workflow.invoke({"messages": [{"role": "user", "content": "Tell me about LangGraph"}]}, config))
print(workflow.invoke({"messages": [{"role": "user", "content": "Now explain checkpointers"}]}, config))
print(workflow.invoke({"messages": [{"role": "user", "content": "What is Langchain"}]}, config))
print(workflow.invoke({"messages": [{"role": "user", "content": "What is Quantum Mechanics"}]}, config))
print(workflow.invoke({"messages": [{"role": "user", "content": "What is Gen AI"}]}, config))
print(workflow.invoke({"messages": [{"role": "user", "content": "What is my name"}]}, config))


snap = workflow.get_state(config)
print("Stored messages after cleanup:", len(snap.values["messages"]))