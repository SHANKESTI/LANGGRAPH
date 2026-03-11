from langgraph.graph import StateGraph,START,END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI(
    model='gpt-4o-mini'
)

class LLMSTATE(TypedDict):
    question:str
    answer:str

Prompt=PromptTemplate(
        template="ANswer the Following question {question}",
        input_variables=['question']
    )

def llm_qa(state:LLMSTATE)->LLMSTATE:
    formatted_prompt = Prompt.format(question=state["question"])   
    answer=model.invoke(formatted_prompt).content
    state['answer']=answer
    return state

graph=StateGraph(LLMSTATE)
graph.add_node('llm_qa',llm_qa)

graph.add_edge(START,'llm_qa')
graph.add_edge('llm_qa',END)

workflow=graph.compile()

intial_state={"question":"Write a detailed essay in Virat Kohli"}
final_state=workflow.invoke(intial_state)

print(final_state)