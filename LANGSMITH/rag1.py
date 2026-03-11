import os
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnableLambda,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "RAG1"

model=ChatOllama(
    model='phi3'
)

loader=PyPDFLoader("D:/LANGGRAPH/LANGSMITH/Hands On Machine Learning with Scikit Learn and TensorFlow.pdf")
docs=loader.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=300
)

splits=splitter.split_documents(docs)

emb=OpenAIEmbeddings(model='text-embedding-3-small')
vs=FAISS.from_documents(splits,emb)

retriever=vs.as_retriever(search_type='similarity',search_kwargs={"k":4})

prompt=ChatPromptTemplate.from_messages([
    ("system","Answer ONLY from the provided context. If not found, say you don't know."),
    ("user","Question:{question} \n \n Context:{context}")

])

def format_docs(docs):
    text = "\n\n".join(d.page_content for d in docs)
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    return text

parallel=RunnableParallel({
    "context":retriever | RunnableLambda(format_docs),
    "question":RunnablePassthrough()

})
parser=StrOutputParser()
chain=parallel | prompt | model | parser

print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip())
print("\nA:", ans)