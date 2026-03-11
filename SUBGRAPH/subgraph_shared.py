from typing import TypedDict
from langgraph.graph import StateGraph,START,END
from langchain_ollama import ChatOllama

model=ChatOllama(
    model='phi3'
)

class ParentState(TypedDict):
    question:str
    answer_eng:str
    answer_hin:str

def Translate_text(state:ParentState):
    answer=state['answer_eng'].strip()
    prompt=f"Translate the following text to Hindi, Keep it clear and neutral Do not add extra content {answer}"
    answer_hin=model.invoke(prompt).content
    return {"answer_hin":answer_hin}

sub=StateGraph(ParentState)
sub.add_node('Translate_text',Translate_text)
sub.add_edge(START,'Translate_text')
sub.add_edge('Translate_text',END)

subgraph=sub.compile()


def generate_text(state:ParentState):
    ques=state['question'].strip()
    prompt=f"You are helpful assistant and answer the question carefully {ques}"
    answer_en=model.invoke(prompt).content
    return {"answer_eng":answer_en}


parent=StateGraph(ParentState)

parent.add_node('generate_text',generate_text)
parent.add_node('subgraph',subgraph)

parent.add_edge(START,'generate_text')
parent.add_edge('generate_text','subgraph')
parent.add_edge('subgraph',END)

graph=parent.compile()

output=graph.invoke({
    "question":"Who is Virat Kohli"
})
print(output)
