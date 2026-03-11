from langgraph.graph import StateGraph,START,END,MessagesState
from langchain_ollama import ChatOllama
from langchain_core.messages.utils import trim_messages,count_tokens_approximately
from langgraph.checkpoint.memory import InMemorySaver

MAX_TOKENS=150

model=ChatOllama(
    model='phi3'
)

def call_model(state:MessagesState):
    messages=trim_messages(
        state['messages'],
        strategy='last',
        max_tokens=MAX_TOKENS,
        token_counter=count_tokens_approximately
    )
    print("Current Token COunt ->",count_tokens_approximately(messages=messages))

    for message in messages:
        print(message.content)

    response=model.invoke(messages)
    return {"messages":[response]}

graph=StateGraph(MessagesState)
graph.add_node("call_model",call_model)

graph.add_edge(START,'call_model')
graph.add_edge("call_model",END)

checkpointer=InMemorySaver()
workflow=graph.compile(checkpointer)

config = {"configurable": {"thread_id": "chat-1"}}
result = workflow.invoke(
    {"messages": [{"role": "user", "content": "Hi, my name is Nitish."}]},
    config,
)

result["messages"][-1].contentresult = workflow.invoke(
    {"messages": [{"role": "user", "content": "I am learning LangGraph."}]},
    config,
)

result["messages"][-1].content
result = workflow.invoke(
    {"messages": [{"role": "user", "content": "Can you explain short term memory?"}]},
    config,
)

result["messages"][-1].content
result = workflow.invoke(
    {"messages": [{"role": "user", "content": "What is my name?"}]},
    config,
)

result["messages"][-1].content
for item in workflow.get_state({"configurable": {"thread_id": "chat-1"}}).values['messages']:
    print(item.content)
    print('-'*120)

