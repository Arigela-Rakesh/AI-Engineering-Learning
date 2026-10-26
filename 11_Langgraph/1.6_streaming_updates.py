
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama


# 1. Define the shared state
class WorkflowState(TypedDict):
    question: str
    answer: str


# 2. Initialize the local model
chat_model = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# 3. Define the chatbot node
def chatbot(state: WorkflowState):
    question = state["question"]

    response = chat_model.invoke(question)

    return {
        "answer": response.content
    }


# 4. Define the summarize_answer node
def summarize_answer(state: WorkflowState):
    answer = state["answer"]

    summary_prompt = (
        "Summarize the following answer in 2 short sentences.\n"
        "Keep the main information.\n\n"
        f"Answer:\n{answer}"
    )

    summary_response = chat_model.invoke(summary_prompt)

    return {
        "answer": summary_response.content
    }


# 5. Create the graph
graph_builder = StateGraph(WorkflowState)


# 6. Add nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("summarize_answer", summarize_answer)


# 7. Define the workflow
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "summarize_answer")
graph_builder.add_edge("summarize_answer", END)


# 8. Compile the graph
graph = graph_builder.compile()


# 9. Define the input
input_data = {
    "question": "Explain LangGraph in simple words.",
    "answer": ""
}


# 10. Stream graph updates
print("Streaming graph updates:\n")

for update in graph.stream(
    input_data,
    stream_mode="updates"
):
    print(update)


# 11. Stream the complete state
print("\nStreaming complete state:\n")

for state in graph.stream(
    input_data,
    stream_mode="values"
):
    print(state)
