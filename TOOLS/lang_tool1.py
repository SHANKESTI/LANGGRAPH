from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage,BaseMessage
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
import requests
import os
import sqlite3
from dotenv import load_dotenv
load_dotenv()


model=ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0,
    max_tokens=300
)

class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

search_tool=DuckDuckGoSearchRun(region='us-en')
@tool
def calculator(first_num:int,second_num:int,operation:str)->dict:
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
def get_weather_data(city:str):
    """Fetch the current weather details for a particular city"""
    url=f'https://api.weatherstack.com/current?access_key=cdf5d42542b5ac1109d04090602ff66d&query={city}'
    response=requests.get(url)
    return response.json()

@tool
def get_stock_price(symbol:str):
    """Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url=f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=HVGLUO19CFJ5U055"
    response=requests.get(url)
    return response.jsnon()


tools=[search_tool,calculator,get_weather_data,get_stock_price]

model_tools=model.bind_tools(tools)

def chat_node(state:ChatState):
    messages=state['messages']
    output=model_tools.invoke(messages)
    return {"messages":[output]}

conn=sqlite3.connect(database='AGENT1.db',check_same_thread=False)
checkpointer=SqliteSaver(conn=conn)

tool_node=ToolNode(tools)
graph=StateGraph(ChatState)
graph.add_node('chat_node',chat_node)
graph.add_node('tools',tool_node)

graph.add_edge(START,'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools','chat_node')

chatbot=graph.compile(checkpointer=checkpointer)
config={"configurable":{"thread_id":'user 1'}}

output=chatbot.invoke(
    {'messages':HumanMessage(content="Can you tell me the current weather condition of the Belagavi city")},
    config=config
)

print(output['messages'][-1].content)