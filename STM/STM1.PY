from langgraph.graph import StateGraph,START,END,MessagesState
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_ollama import ChatOllama

model=ChatOllama(
    model='phi3'
)

def call_node(state:MessagesState):
    response=model.invoke(state['messages'])
    return {"messages":[response]}

graph=StateGraph(MessagesState)

graph.add_node('call_node',call_node)

graph.add_edge(START,'call_node')
graph.add_edge('call_node',END)

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres"




with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    workflow=graph.compile(checkpointer=checkpointer)

    t1 = {"configurable": {"thread_id": "thread-1"}}
    workflow.invoke({"messages": [{"role": "user", "content": "Hi, my name is THE SHAN KESTI"}]}, t1)
    out1 = workflow.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, t1)
    print("Thread-1:", out1["messages"][-1].content)