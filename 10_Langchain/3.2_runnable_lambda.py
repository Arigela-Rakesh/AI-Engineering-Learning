
from langchain_core.runnables import RunnableLambda

# Define a normal Python function.
def format_topic(topic: str) -> str:
    return "Explain this AI topic: " + topic

# Convert the function into a LangChain Runnable.
topic_runnable = RunnableLambda(format_topic)

# Execute the Runnable.
result = topic_runnable.invoke("Retrieval-Augmented Generation")

print(result)
