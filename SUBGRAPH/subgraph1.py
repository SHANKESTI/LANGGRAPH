from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph,START,END
from typing import TypedDict

model=ChatOllama(
    model='phi3'
)
class substate(TypedDict):
    input_text:str
    translate_text:str

def translate1_text(state:substate):
    input_text=state['input_text'].strip()
    prompt=f"Convert the given english text to hindi {input_text}"
    output=model.invoke(prompt).content
    return {'translate_text':output}


sub=StateGraph(substate)
sub.add_node('translate1_text',translate1_text)
sub.add_edge(START,'translate1_text')
sub.add_edge('translate1_text',END)
subgraph=sub.compile()

class parentclass(TypedDict):
    question:str
    answer_en:str
    answer_hn:str


def generate_text(state:parentclass):
    question=state['question']
    prompt=f"Generate a detailed review about the {question}"
    output=model.invoke(prompt).content
    return {"answer_en":output}

def translate_text(state:parentclass):
    answer_en=state['answer_en']
    output=subgraph.invoke({
    "input_text": answer_en
    })
    return {"answer_hn":output['translate_text']}



parent=StateGraph(parentclass)
parent.add_node('generate_text',generate_text)
parent.add_node('translate_text',translate_text)

parent.add_edge(START,'generate_text')
parent.add_edge('generate_text','translate_text')
parent.add_edge('translate_text',END)

graph=parent.compile()

output=graph.invoke({"question":"Virat Kohli"})

print(output)