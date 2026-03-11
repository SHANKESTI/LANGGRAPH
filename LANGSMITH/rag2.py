import os
from langsmith import traceable
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from dotenv import load_dotenv
load_dotenv()

model=ChatOllama(
    model='phi3'
)
os.environ['LANGCHAIN_PROJECT']='RAG2'
path="D:/LANGGRAPH/LANGSMITH/Hands On Machine Learning with Scikit Learn and TensorFlow.pdf"
@traceable(name="load_pdf")
def load_pdf(path:str):
    loader=PyPDFLoader(path)
    return loader.load()

@traceable(name="text_splitter")
def text_splitter(docs,chunk_size=1000,chunk_overlap=300):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    emd=OpenAIEmbeddings(model="text-embedding-3-small")
    vs=FAISS.from_documents(splits,emd)
    return vs

@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path:str):
    docs=load_pdf(pdf_path)
    splits=text_splitter(docs)
    vs=build_vectorstore(splits)
    return vs

prompt=ChatPromptTemplate.from_messages([
    ("system","Answer ONLY from the provided context. If not found, say you don't know."),
    ("human","Question: {question} \n\n Context:\n{context}")
])

def format_docs(docs):
    text = "\n\n".join(d.page_content for d in docs)
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    return text
vector_store=setup_pipeline(path)
retriever=vector_store.as_retriever(search_type='similarity',search_kwargs={'k':4})

parallel=RunnableParallel({
    "context":retriever|RunnableLambda(format_docs),
    "question":RunnablePassthrough()
})

chain=parallel | prompt | model | StrOutputParser()

print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ").strip()


config = {
    "run_name": "pdf_rag_query"
}

ans = chain.invoke(q, config=config)
print("\nA:", ans)