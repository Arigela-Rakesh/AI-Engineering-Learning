from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END


# 1. Define the shared state with a reducer
class WorkflowState(TypedDict):
    question: str
    messages: Annotated[list[str], operator.add]


# 2. Identify the topic
def identify_topic(state: WorkflowState):
    question = state["question"].lower()

    if "python" in question:
        topic = "Python"
    elif "langgraph" in question:
        topic = "LangGraph"
    else:
        topic = "General"

    return {
        "messages": [f"Identified topic: {topic}"]
    }


# 3. Analyze the intent
def analyze_intent(state: WorkflowState):
    question = state["question"].lower()

    if "how" in question:
        intent = "Learning"
    elif "what" in question:
        intent = "Information"
    else:
        intent = "General inquiry"

    return {
        "messages": [f"Identified intent: {intent}"]
    }

def extract_keywords(state: WorkflowState):
    return {
        "messages": ["Extracted keywords: Python, AI"]
    }
    
# 4. Build the graph
graph_builder = StateGraph(WorkflowState)

graph_builder.add_node("identify_topic", identify_topic)
graph_builder.add_node("analyze_intent", analyze_intent)
graph_builder.add_node("extract_keywords", extract_keywords)

graph_builder.add_edge(START, "identify_topic")

# Both nodes execute after START
graph_builder.add_edge(START, "analyze_intent")

graph_builder.add_edge(START, "extract_keywords")


graph_builder.add_edge("identify_topic", END)
graph_builder.add_edge("analyze_intent", END)
graph_builder.add_edge("extract_keywords", END)

graph = graph_builder.compile()


# 5. Run the graph
input_data = {
    "question": "How can I use Python in AI?",
    "messages": []
}

result = graph.invoke(input_data)

print("\nFinal Workflow Result:\n")
print("Question:", result["question"])

for message in result["messages"]:
    print(message)