from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from dotenv import load_dotenv
load_dotenv()

model=ChatOpenAI(
    model="gpt-4o-mini"
)

class Blogstate(TypedDict):
    title:str
    outline:str
    blog:str

prompt1=PromptTemplate(
    template="Generate a detailed outline for a blog on the topic {title}",
    input_variables=['title']
)

prompt2=PromptTemplate(
    template="Write a detailed blog on the title - {title} using the follwing outline \n {outline}",
    input_variables=['title','outline']
)

def create_outline(state:Blogstate)->Blogstate:
    title=state['title']
    prompt=prompt1.format(title=title)
    outline=model.invoke(prompt).content 
    state['outline']=outline
    return state

def create_blog(state:Blogstate)->Blogstate:
    title=state['title']
    outline=state['outline']
    prompt=prompt2.format(title=title,outline=outline)
    blog=model.invoke(prompt).content
    state['blog']=blog
    return state


graph=StateGraph(Blogstate)

graph.add_node('create_outline',create_outline)
graph.add_node('create_blog',create_blog)

graph.add_edge(START,'create_outline')
graph.add_edge('create_outline','create_blog')
graph.add_edge('create_blog',END)

workflow=graph.compile()

initial_state={"title":"Rise Of AI in India"}

final_state=workflow.invoke(initial_state)

print(final_state)


png_data = workflow.get_graph().draw_mermaid_png()

with open("workflow_graph.png", "wb") as f:
    f.write(png_data)

print("Graph saved as workflow_graph.png")