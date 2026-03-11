from langgraph.graph import START,END,StateGraph
from langchain_ollama import ChatOllama
from typing import TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
import sqlite3
from dotenv import load_dotenv
load_dotenv()

model=ChatOllama(
    model='phi3'
)
model1=ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0,
    max_tokens=300
)

class JOKESTATE(TypedDict):
    topic:str
    joke:str
    explaination:str


def generate_joke(state:JOKESTATE):
    prompt=f"Generate a joke on the topic {state['topic']}"
    output=model1.invoke(prompt).content
    return {"joke":output}

def explain_joke(state:JOKESTATE):
    prompt=f"Explain the following joke {state['joke']}"
    output=model1.invoke(prompt).content
    return {"explaination":output}

conn=sqlite3.connect(database='shan.db',check_same_thread=False)

checkpointer=SqliteSaver(conn=conn)

graph=StateGraph(JOKESTATE)

graph.add_node('generate_joke',generate_joke)
graph.add_node('explain_joke',explain_joke)
graph.add_edge(START,'generate_joke')
graph.add_edge('generate_joke','explain_joke')
graph.add_edge('explain_joke',END)

workflow=graph.compile(checkpointer=checkpointer)

config={"configurable":{"thread_id":"user1"}}

initial_state={'topic':"Pizza"}

for chunk in workflow.stream(initial_state,config=config):
    print(chunk)
