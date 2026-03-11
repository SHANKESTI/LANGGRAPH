from langgraph.graph import StateGraph,START,END
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from typing import TypedDict,Annotated
from langchain_openai import ChatOpenAI
import os
import requests
import sqlite3
from dotenv import load_dotenv
load_dotenv()
os.environ['LANGCHAIN_PROJECT']="AGENT_TOOL"
model=ChatOllama(
    model='llama3.1'
)
model1=ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0,
    max_tokens=300
)

search_tool=DuckDuckGoSearchRun(region='us-en')

@tool
def calculator(first_num:float,second_num:float,operation:str)->dict:
    """
     Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}
    

@tool
def get_stock_price(symbol:str)->dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.

    """
    url=f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=HVGLUO19CFJ5U055"
    r=requests.get(url)
    return r.json()


@tool
def get_weather_data(city:str)->str:
    """
    This function featches the current weather data for a given city
    """
    url=f'https://api.weatherstack.com/current?access_key=cdf5d42542b5ac1109d04090602ff66d&query={city}'
    response=requests.get(url)
    return response.json()


tools=[search_tool,calculator,get_stock_price,get_weather_data]

model_tools=model.bind_tools(tools)
model1_tools=model1.bind_tools(tools)

class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

def chat_node(state:ChatState):
    """LLM node that may answer or request a tool call"""
    messages=state['messages']
    response=model1_tools.invoke(messages)
    return {"messages":[response]}

tool_node=ToolNode(tools)

conn=sqlite3.connect(database='AGENT.db',check_same_thread=False)

checkpointer=SqliteSaver(conn=conn)

graph=StateGraph(ChatState)
graph.add_node('chat_node',chat_node)
graph.add_node('tools',tool_node)

graph.add_edge(START,"chat_node")
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools','chat_node')

chatbot=graph.compile(checkpointer=checkpointer)
config={"configurable":{"thread_id":"user-1"}}

output=chatbot.invoke({
    "messages":[
        HumanMessage(content="Give me the current weather condition of belagavi city")
    ]
},config=config)

print(output["messages"][-1].content)




