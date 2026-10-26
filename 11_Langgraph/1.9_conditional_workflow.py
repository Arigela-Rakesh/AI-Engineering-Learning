from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama


# 1. Define the shared state
class WorkflowState(TypedDict):
    question: str
    topic: str
    answer: str


# 2. Initialize the model
chat_model = ChatOllama(model="llama3.2:3b", temperature=0)


# 3. Identify the topic
def identify_topic(state: WorkflowState):
    question = state["question"].lower()

    if "langgraph" in question:
        topic = "LangGraph"
    elif "python" in question:
        topic = "Python"
    else:
        topic = "General"

    return {"topic": topic}


# 4. Generate a LangGraph answer
def langgraph_answer(state: WorkflowState):
    return {
        "answer": (
            "LangGraph is a framework for building "
            "stateful workflows and AI agents. "
            "It allows you to connect nodes, manage state, "
            "and control execution using graph-based logic."
        )
    }


# 5. Generate a Python answer
def python_answer(state: WorkflowState):
    return {
        "answer": (
            "Python is a high-level programming language "
            "known for its readable syntax. It is widely "
            "used in web development, automation, and AI."
        )
    }


# 6. Generate a general answer
def general_answer(state: WorkflowState):
    response = chat_model.invoke(state["question"])

    return {"answer": response.content}


# 7. Define the routing function
def route_question(state: WorkflowState):
    topic = state["topic"]

    if topic == "LangGraph":
        return "langgraph"

    elif topic == "Python":
        return "python"

    else:
        return "general"


# 8. Build the graph
graph_builder = StateGraph(WorkflowState)

graph_builder.add_node("identify_topic", identify_topic)
graph_builder.add_node("langgraph_answer", langgraph_answer)
graph_builder.add_node("python_answer", python_answer)
graph_builder.add_node("general_answer", general_answer)

graph_builder.add_edge(START, "identify_topic")

graph_builder.add_conditional_edges(
    "identify_topic",
    route_question,
    {
        "langgraph": "langgraph_answer",
        "python": "python_answer",
        "general": "general_answer",
    },
)

graph_builder.add_edge("langgraph_answer", END)
graph_builder.add_edge("python_answer", END)
graph_builder.add_edge("general_answer", END)

graph = graph_builder.compile()


# 9. Run the graph
input_data = {"question": "What is the capital of India?", "topic": "", "answer": ""}

result = graph.invoke(input_data)

print("\nFinal Workflow Result:\n")
print("Question:", result["question"])
print("Topic:", result["topic"])
print("Answer:", result["answer"])
