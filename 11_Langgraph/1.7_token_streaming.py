from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama


# 1. Define the shared state
class WorkflowState(TypedDict):
    question: str
    answer: str


# 2. Initialize the chat model
chat_model = ChatOllama(model="llama3.2:3b", temperature=0)


# 3. Define the chatbot node
def chatbot(state: WorkflowState):
    question = state["question"]

    response = chat_model.invoke(question)

    return {"answer": response.content}


# 4. Build the graph
graph_builder = StateGraph(WorkflowState)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()


# 5. Prepare the input
input_data = {"question": "What is the difference between LangChain and LangGraph?", "answer": ""}


# 6. Stream LLM message chunks
print("Streaming LLM response:\n")

for message_chunk, metadata in graph.stream(input_data, stream_mode="messages"):
    content = message_chunk.content

    if isinstance(content, str):
        print(content, end="", flush=True)

print("\n\nStreaming completed.")
