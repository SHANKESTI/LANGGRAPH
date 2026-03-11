from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

prompt1=PromptTemplate(
    template="Generate a detailed report on the topic {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="Generate a 5 line summary on the following text {text}",
    input_variables=['text']
)

parser=StrOutputParser()

workflow=prompt1 | model | parser | prompt2 | model | parser

for chunk in workflow.stream({"topic":"Virat Kohli"}):
    print(chunk, end="", flush=True)