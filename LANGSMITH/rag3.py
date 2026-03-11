import os
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langsmith import traceable
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
load_dotenv()
model=ChatOllama(
    model='phi3'
)

os.environ["LANGCHAIN_PROJECT"]="RAG3"
path="D:/LANGGRAPH/LANGSMITH/Hands On Machine Learning with Scikit Learn and TensorFlow.pdf"
@traceable(name="document_loader")
def document_loader(path:str):
    loader=PyPDFLoader(path)
    return loader.load()

@traceable(name="text_splitter")
def text_splitter(docs,chunk_size=1000,chunk_overlap=300):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300
    )
    return splitter.split_documents(docs)

@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    emb=OpenAIEmbeddings(model='text-embedding-3-small')
    return FAISS.from_documents(splits,emb)

prompt=ChatPromptTemplate.from_messages([
    ("system","Answer the question from the given pdf If you dont know say dont know"),
    ("human","Question:{question}\n\n Context:{context}")
])

@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path:str,chunk_size=1000,chunk_overlap=300):
    docs=document_loader(pdf_path)
    splits=text_splitter(docs,chunk_size,chunk_overlap)
    vs=build_vectorstore(splits)
    return vs

def format_docs(docs):
    text = "\n\n".join(d.page_content for d in docs)
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    return text

@traceable(name="pdf_rag_full_run")
def setup_pipeline_and_query(pdf_path: str, question: str):
    vectorstore = setup_pipeline(pdf_path, chunk_size=1000, chunk_overlap=150)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })

    chain = parallel | prompt | model | StrOutputParser()

    lc_config = {"run_name": "pdf_rag_query"}
    return chain.invoke(question, config=lc_config)

if __name__ == "__main__":
    print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
    q = input("\nQ: ").strip()
    ans = setup_pipeline_and_query(path, q)
    print("\nA:", ans)



    