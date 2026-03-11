from langgraph.graph import StateGraph,START,END
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict

class substate(TypedDict):
    input_text:str
    translate_text:str

model=ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0,
    max_tokens=300
)

def translate_text(state:substate):
    prompt=f"""
    Translate the following text to Hindi.
    Keep it natural and clear. Do not add extra content.
    Text:
    {state["input_text"]}""".strip()
    output=model.invoke(prompt).content
    return {'translate_text':output}

subgraph=StateGraph(substate)
subgraph.add_node('translate_text',translate_text)
subgraph.add_edge(START,'translate_text')
subgraph.add_edge('translate_text',END)
sub=subgraph.compile()

class ParentState(TypedDict):
    question:str
    answer_eng:str
    answer_hin:str

def generate_answer(state:ParentState):   
   answer = model.invoke(f"You are a helpful assistant. Answer clearly.\n\nQuestion: {state['question']}").content
   return {'answer_eng': answer}

def translate_answer(state:ParentState):
    result=sub.invoke({'input_text':state['answer_eng']})
    return {'answer_hin':result['translate_text']}

parent=StateGraph(ParentState)
parent.add_node('generate_answer',generate_answer)
parent.add_node('translate_answer',translate_answer)
parent.add_edge(START,'generate_answer')
parent.add_edge('generate_answer','translate_answer')
parent.add_edge('translate_answer',END)

graph=parent.compile()

png_data = graph.get_graph().draw_mermaid_png()

with open("subgraph.png", "wb") as f:
    f.write(png_data)

result=graph.invoke({'question': 'What is quantum physics'})
print(result)