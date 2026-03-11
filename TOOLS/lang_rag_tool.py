from langgraph.graph import StateGraph,START,END
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from typing import TypedDict,Annotated
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from langchain_core.messages import HumanMessage,AIMessage,BaseMessage
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnablePassthrough
import os
from dotenv import load_dotenv
load_dotenv()

model=ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=300
)
class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]


path="D:/LANGGRAPH/LANGSMITH/Hands On Machine Learning with Scikit Learn and TensorFlow.pdf"

def load_document(path:str):
    loader=PyPDFLoader(path)
    return loader.load()

def docs_splitter(docs:str,chunk_size=1000,chunk_overlap=300):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def build_vectorstore(split:str):
    emb=OpenAIEmbeddings(model='text-embedding-3-small')
    vectorstore=FAISS.from_documents(split,emb)
    return vectorstore.as_retriever(search_type='similarity',search_kwargs={"k":4})

def set_pipeline(path:str,chunk_size=1000,chunk_overlap=300):
    docs=load_document(path)
    split=docs_splitter(docs,chunk_size,chunk_overlap)
    vs=build_vectorstore(split)
    return vs

def format_docs(docs):
    text = "\n\n".join(d.page_content for d in docs)
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    return text
retriever=set_pipeline(path)

@tool
def rag_tool(query: str) -> str:
    """Use this tool to answer questions from the uploaded PDF."""
    
    docs = retriever.get_relevant_documents(query)
    
    context =format_docs(docs)
    prompt = f"""
    Use the following context to answer the question.
    
    Context:
    {context}
    
    Question:
    {query}
    """
    prompt = prompt    
    
    response = model.invoke(prompt)
    
    return response.content


Tools=[rag_tool]
model_tools=model.bind_tools(Tools)

tool_node=ToolNode(Tools)

def chat_node(state:ChatState):
    messages=state['messages']
    output=model_tools.invoke(messages)
    return {"messages":[output]}
graph=StateGraph(ChatState)
graph.add_node('chat_node',chat_node)
graph.add_node('tools',tool_node)

graph.add_edge(START,'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools','chat_node')
graph.add_edge("chat_node", END)

chatbot=graph.compile()

output=chatbot.invoke({
    "messages":[HumanMessage(content="What is mean by SVM explain it ?")]
})

print(output['messages'][-1].content)






