from langgraph.graph import StateGraph,START,END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,BaseMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode,tools_condition
from typing import TypedDict,Annotated
from langgraph.types import Command,interrupt
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import requests,os
load_dotenv()
os.environ['LANGCHAIN_PROJECT']="CHATBOT_HITL"

model=ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0,
    max_tokens=300
)

search=DuckDuckGoSearchRun(region='us-en')
@tool
def websearch(query:str):
    """
    Perform a web search using DuckDuckGo and return the most relevant results.

    Args:
        query (str): The search query string to look up online.

    Returns:
        str: A text summary or snippet of the search results related to the query.
    """

    return search.run(query)


@tool
def get_stock_price(symbol:str):
    """    
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """

    url=f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=HVGLUO19CFJ5U055"
    response=requests.get(url)
    return response.json()

@tool
def purchase_stock(symbol:str,quantity:int)->dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
    decision=interrupt(f"Approve buying of {quantity} of {symbol} by (yes/no)\n")
    if isinstance(decision,str) and decision.lower().strip() =='yes':
        return {
            "status":"Success",
            "message":f"Purchase order placed for {quantity} shares of {symbol}.",
            "quantity":quantity,
            "symbol":symbol
        }
    else:
        return {
            "status":"Cancelled",
            "message":f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "quantity":quantity,
            "symbol":symbol
        }

tools=[websearch,get_stock_price,purchase_stock]

model_tools=model.bind_tools(tools)

class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

def chat_node(state:ChatState):
    messages=state['messages']
    output=model_tools.invoke(messages)
    return {"messages":[output]}

tool_node=ToolNode(tools)

graph=StateGraph(ChatState)
graph.add_node('chat_node',chat_node)
graph.add_node('tools',tool_node)

graph.add_edge(START,'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools','chat_node')
graph.add_edge('chat_node',END)

checkpointer=InMemorySaver()
config={"configurable":{"thread_id":"user-1"}}

chatbot=graph.compile(checkpointer=checkpointer)
png_data = chatbot.get_graph().draw_mermaid_png()

with open("chatbot.png", "wb") as f:
    f.write(png_data)

if __name__=="__main__":
    while True:
        user_input=input("You: ")
        if user_input.lower().strip() in {'exit','quit'}:
            print("Have a Great time....... Bye")
            break
        result=chatbot.invoke({
            "messages":[HumanMessage(content=user_input)]
        },config=config)

        interrupts=result.get("__interrupt__",[])
        if interrupts:
            prompt_to_human=interrupts[0].value
            print(f"HITL: \t {prompt_to_human}")
            decision=input("Your Decision:").strip().lower()

            result=chatbot.invoke(
                Command(resume=decision),
                config=config
            )
        messages=result['messages']
        last_msg=messages[-1]
        print(f"bot : {last_msg.content}\n")
