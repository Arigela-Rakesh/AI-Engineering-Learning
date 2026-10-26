
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
)

def format_topic(topic: str) -> str:
    return "Explain this AI topic: " + topic

chain = (
    RunnablePassthrough()
    | RunnableLambda(format_topic)
)

result = chain.invoke("Vector Databases")

print(result)
