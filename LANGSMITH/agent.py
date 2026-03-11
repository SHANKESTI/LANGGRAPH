from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph,START,END
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent,AgentExecutor
from langchain import hub
from dotenv import load_dotenv
import requests
import os

os.environ['LANGCHAIN_PROJECT']='SIMPLEAGENT'
load_dotenv()

search_tool=DuckDuckGoSearchRun()
@tool
def get_weather_data(city:str)->str:
    """
    This function featches the current weather data for a given city
    """
    url=f'https://api.weatherstack.com/current?access_key=cdf5d42542b5ac1109d04090602ff66d&query={city}'
    response=requests.get(url)
    return response.json()

prompt=hub.pull("hwchase17/react")

model=ChatOllama(
    model="phi3"
)

agent=create_react_agent(
    llm=model,
    tools=[search_tool,get_weather_data],
    prompt=prompt
)

agent_executor=AgentExecutor(
    agent=agent,
    tools=[search_tool,get_weather_data],
    verbose=True,
    max_iterations=3
)
response=agent_executor.invoke({"input":"What is the current temp of gurgaon"})

print(response)

print(response['output'])
