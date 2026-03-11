from langgraph.graph import StateGraph,START,END,MessagesState
from langchain_core.messages import RemoveMessage,HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages.utils import trim_messages,count_tokens_approximately

model=ChatOllama(
    model="phi3"
)
MAX_TOKENS=300
class ChatState(MessagesState):
    summary:str

def chat_node(state:ChatState):
    messages=trim_messages(
        state['messages'],
        strategy='last',
        max_tokens=MAX_TOKENS,
        token_counter=count_tokens_approximately
    )

    if state.get('summary'):
        messages= [HumanMessage(content=state["summary"])] + messages
    
    output=model.invoke(messages)
    return {"messages":[output]}

def delete_old(state:ChatState):
    msgs=state['messages']
    if len(msgs)>20:
        messages_to_remove=msgs[:12]
        return{"messages":[RemoveMessage(id=m.id) for m in messages_to_remove]}
    return {}

def summarize(state:ChatState):
    existing_summary=state['summary']
    if existing_summary:
        prompt=(
            f"Existing summary:\n{existing_summary}\n\n"
            "Extend the summary using the new conversation above."
        )
    else:
        prompt="Summarize the above conversation"
    messages_for_summary=state['messages']+[HumanMessage(content=prompt)]
    response=model.invoke(messages_for_summary)
    return {"summary":response.content}

def should_summarize(state:ChatState):
    if len(state['messages'])>25:
        return "summarize"
    return "delete_old"

graph=StateGraph(ChatState)
graph.add_node("chat_node",chat_node)
graph.add_node("delete_old",delete_old)
graph.add_node("summarize",summarize)

graph.add_edge(START,"chat_node")
graph.add_conditional_edges(
    "chat_node",
    should_summarize,
    {
        "summarize":"summarize",
        "delete_old":"delete_old"
    }
)
graph.add_edge("summarize","delete_old")
graph.add_edge("delete_old",END)

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres"

config = {"configurable": {"thread_id": "t1"}}

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    workflow=graph.compile(checkpointer=checkpointer)
    
    def run_turn(text:str):
        out = workflow.invoke(
            {"messages": [HumanMessage(content=text)], "summary": ""},
            config=config
        )
        return out['messages'][-1].content
    print(run_turn("Quantum Mechanics"))