from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage,AIMessage,BaseMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.types import interrupt,Command
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict,Annotated
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import requests
from dotenv import load_dotenv
load_dotenv()

model=ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0,
    max_tokens=300
)
search_tool=DuckDuckGoSearchRun(region='us-en')
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
    decision=interrupt(f"Approve buying {quantity} shares of {symbol} ? (yes/no)")
    if isinstance(decision,str) and decision.lower()=='yes':
        return {
            'status':'success',
            'message':f'Purchase order placed for {quantity} shares of {symbol}.',
            'symbol':symbol,
            'quantity':quantity,
        }
    else:
        return {
            'status':'cancelled',
            'messages':f"Purchase of {quantity} shares of {symbol} was declined by human.",
            'symbol':symbol,
            'quantity':quantity
        }

tools=[get_stock_price,purchase_stock,search_tool]
model_tools=model.bind_tools(tools)

class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]


def chat_node(state:ChatState):
    """LLM node that may answer or request a tool call"""
    messages=state['messages']
    response=model_tools.invoke(messages)
    return {'messages':[response]}

tool_node=ToolNode(tools)

checkpointer=InMemorySaver()

graph=StateGraph(ChatState)

graph.add_node('chat_node',chat_node)
graph.add_node('tools',tool_node)

graph.add_edge(START,'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools','chat_node')
graph.add_edge('chat_node',END)

chatbot=graph.compile(checkpointer=checkpointer)

if __name__=='__main__':


    thread_id="Demo Thread"

    while True:
        user_input=input("You:")
        if user_input.lower().strip() in {'exit','quit'}:
            print("Bye")
            break
    
        state={"messages":HumanMessage(content=user_input)}
        result=chatbot.invoke(
            state,
            config={"configurable":{"thread_id":thread_id}}
        )

        interrupts=result.get("__interrupt__",[])
        if interrupts:
            prompt_to_human=interrupts[0].value
            print(f"HITL:{prompt_to_human}")
            decision=input("Your Decision:").lower().strip()

            result=chatbot.invoke(
                Command(resume=decision),
                config={"configurable":{"thread_id":thread_id}}
            )
        messages=result['messages']
        last_msg=messages[-1]
        print(f"Bot :{last_msg.content} \n")
        
