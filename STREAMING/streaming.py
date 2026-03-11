from langgraph.graph import StateGraph,START,END
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict

model=ChatOllama(
    model="phi3"
)

class JOKESTATE(TypedDict):
    topic:str
    joke:str
    explaination:str


def generate_joke(state:JOKESTATE):
    prompt=f"Generate a joke on the topic {state['topic']}"
    output=model.invoke(prompt).content
    return {'joke':output}

def explain_joke(state:JOKESTATE):
    prompt=f"Explain the following joke {state['joke']}"
    output=model.invoke(prompt).content
    return {"explaination":output}

graph=StateGraph(JOKESTATE)
graph.add_node('generate_joke',generate_joke)
graph.add_node('explain_joke',explain_joke)

graph.add_edge(START,'generate_joke')
graph.add_edge('generate_joke','explain_joke')
graph.add_edge('explain_joke',END)

checkpointer=InMemorySaver()

workflow=graph.compile(checkpointer=checkpointer)

config={"configurable":{"thread_id":"1"}}

initial_state={"topic":"PIZZA"}

for chunk in workflow.stream(initial_state,config=config):
    print(chunk)