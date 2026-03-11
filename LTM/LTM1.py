from langgraph.store.memory import InMemoryStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

namespace=("user","u1")
store=InMemoryStore()
store.put(namespace,"1",{"data":"User likes Pizza"})
store.put(namespace,"2",{'data':"User prefers dark mode"})

namespace2=("user2","u2")

store.put(namespace2,"1",{"data":"User likes Pasta"})
store.put(namespace2,"2",{"data":"User likes python language"})

print(store.get(namespace,"1"))
items=store.search(namespace2)
for items in items:
    print(items.value)

emd=OpenAIEmbeddings(
    model="text-embedding-3-small"
)

store=InMemoryStore(index={'embed':emd,'dims':1536})

store.put(namespace, "1", {"data": "User prefers concise answers over long explanations"})
store.put(namespace, "2", {"data": "User likes examples in Python"})
store.put(namespace, "3", {"data": "User usually works late at night"})
store.put(namespace, "4", {"data": "User prefers dark mode in applications"})
store.put(namespace, "5", {"data": "User is learning machine learning"})
store.put(namespace, "6", {"data": "User dislikes overly theoretical explanations"})
store.put(namespace, "7", {"data": "User prefers step-by-step reasoning"})
store.put(namespace, "8", {"data": "User is based in India"})
store.put(namespace, "9", {"data": "User likes real-world analogies"})
store.put(namespace, "10", {"data": "User prefers bullet points over paragraphs"})

print(store.search(namespace,query="what is user learning",limit=1))

items=store.search(namespace,query="What are the users preference?",limit=3)
for item in items:
    print(item.value)