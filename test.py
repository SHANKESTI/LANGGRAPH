# test.py

import sys

print("=" * 50)
print("Python Version:", sys.version)
print("=" * 50)

# LangChain
try:
    import langchain
    print("LangChain Version:", langchain.__version__)
except ImportError:
    print("LangChain not installed")

# LangChain Core
try:
    import langchain_core
    print("LangChain Core Version:", langchain_core.__version__)
except ImportError:
    print("LangChain Core not installed")

# LangGraph
try:
    import langgraph
    print("LangGraph Version:", langgraph.__version__)
except ImportError:
    print("LangGraph not installed")

# LangChain OpenAI
try:
    import langchain_openai
    print("LangChain OpenAI Version:", langchain_openai.__version__)
except ImportError:
    print("LangChain OpenAI not installed")

# LangChain Community
try:
    import langchain_community
    print("LangChain Community Version:", langchain_community.__version__)
except ImportError:
    print("LangChain Community not installed")

# LangChain Experimental
try:
    import langchain_experimental
    print("LangChain Experimental Version:", langchain_experimental.__version__)
except ImportError:
    print("LangChain Experimental not installed")

# LangGraph Checkpoint
try:
    import langgraph.checkpoint
    print("LangGraph Checkpoint Available")
except ImportError:
    print("LangGraph Checkpoint not installed")

print("=" * 50)
print("Test completed")
print("=" * 50)