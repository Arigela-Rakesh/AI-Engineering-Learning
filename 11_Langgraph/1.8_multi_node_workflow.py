from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama


# 1. Define the shared state
class WorkflowState(TypedDict):
    question: str
    topic: str
    answer: str
    validation: str
    source: str


# 2. Initialize the model
chat_model = ChatOllama(model="llama3.2:3b", temperature=0)


# 3. Identify the topic
def identify_topic(state: WorkflowState):
    question = state["question"]

    prompt = (
        "Identify the main topic of the question. "
        "Return only a short topic label.\n\n"
        f"Question: {question}"
    )

    response = chat_model.invoke(prompt)

    return {"topic": response.content.strip()}


def identify_source(state: WorkflowState):
    return {"source": "LangGraph documentation"}


# 4. Generate the answer
def generate_answer(state: WorkflowState):
    question = state["question"]
    topic = state["topic"]

    prompt = (
        "Answer the user's question in simple words. "
        "If you are unsure, say so. Do not invent facts.\n\n"
        f"Topic: {topic}\n"
        f"Question: {question}"
    )

    response = chat_model.invoke(prompt)

    return {"answer": response.content}


# 5. Validate the answer
def validate_answer(state: WorkflowState):
    answer = state["answer"]

    if answer.strip():
        validation = "Answer generated successfully."
    else:
        validation = "Answer is empty."

    return {"validation": validation}


# 6. Build the graph
graph_builder = StateGraph(WorkflowState)

graph_builder.add_node("identify_topic", identify_topic)
graph_builder.add_node("identify_source", identify_source)
graph_builder.add_node("generate_answer", generate_answer)
graph_builder.add_node("validate_answer", validate_answer)


graph_builder.add_edge(START, "identify_topic")
graph_builder.add_edge("identify_topic", "identify_source")
graph_builder.add_edge("identify_source", "generate_answer")
graph_builder.add_edge("generate_answer", "validate_answer")
graph_builder.add_edge("validate_answer", END)

graph = graph_builder.compile()


# 7. Run the graph
input_data = {
    "question": "What is LangGraph used for?",
    "topic": "",
    "source": "",
    "answer": "",
    "validation": "",
}

result = graph.invoke(input_data)

print("\nFinal Workflow Result:\n")
print("Question:", result["question"])
print("Topic:", result["topic"])
print("Source:", result["source"])
print("Answer:", result["answer"])
print("Validation:", result["validation"])
