from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# 1. Define the shared state
class WorkflowState(TypedDict):
    question: str
    topic: str
    answer: str


# 2. Classify the question
def classify_question(state: WorkflowState):
    question = state["question"].lower()

    if "ai" in question or "artificial intelligence" in question:
        topic = "AI"

    elif "python" in question:
        topic = "Python"

    elif "javascript" in question or "react" in question:
        topic = "JavaScript"

    else:
        topic = "General"

    print("Identified topic:", topic)

    return {"topic": topic}


# 3. Define the AI node
def ai_node(state: WorkflowState):
    return {
        "answer": "AI is the field of building systems that perform intelligent tasks."
    }


# 4. Define the Python node
def python_node(state: WorkflowState):
    return {"answer": "Python is a popular programming language."}


def javascript_node(state: WorkflowState):
    return {
        "answer": "JavaScript is a programming language used to build interactive web applications."
    }


# 5. Define the general node
def general_node(state: WorkflowState):
    return {"answer": "I can provide a general answer to your question."}


# 6. Define the routing function
def route_question(state: WorkflowState):
    topic = state["topic"]

    if topic == "AI":
        return "ai"

    elif topic == "Python":
        return "python"

    elif topic == "JavaScript":
        return "javascript"

    else:
        return "general"


# 7. Create the graph
graph_builder = StateGraph(WorkflowState)


# 8. Add nodes
graph_builder.add_node("classify_question", classify_question)
graph_builder.add_node("ai_node", ai_node)
graph_builder.add_node("python_node", python_node)
graph_builder.add_node("javascript_node", javascript_node)
graph_builder.add_node("general_node", general_node)


# 9. Add normal edge
graph_builder.add_edge(START, "classify_question")


# 10. Add conditional edges
graph_builder.add_conditional_edges(
    "classify_question",
    route_question,
    {
        "ai": "ai_node",
        "python": "python_node",
        "javascript": "javascript_node",
        "general": "general_node",
    },
)


# 11. Connect all nodes to END
graph_builder.add_edge("ai_node", END)
graph_builder.add_edge("python_node", END)
graph_builder.add_edge("javascript_node", END)
graph_builder.add_edge("general_node", END)


# 12. Compile the graph
graph = graph_builder.compile()


# 13. Execute the graph
result = graph.invoke({"question": "What is React?", "topic": "", "answer": ""})


# 14. Display the result
print("\nFinal result:")
print(result)
