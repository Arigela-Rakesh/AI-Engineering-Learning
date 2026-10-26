from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# 1. Define the shared state
class WorkflowState(TypedDict):
    question: str
    topic: str
    intent: str
    keywords: list[str]
    final_result: str


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
        "topic": topic
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
        "intent": intent
    }


# 4. Extract keywords
def extract_keywords(state: WorkflowState):
    question = state["question"].lower()

    keywords = []

    if "python" in question:
        keywords.append("Python")

    if "langgraph" in question:
        keywords.append("LangGraph")

    if "ai" in question:
        keywords.append("AI")

    if len(keywords) == 0:
        keywords.append("General")

    return {
        "keywords": keywords
    }


# 5. Combine the results
def combine_results(state: WorkflowState):
    topic = state["topic"]
    intent = state["intent"]
    keywords = state["keywords"]

    final_result = (
        f"Topic: {topic}\n"
        f"Intent: {intent}\n"
        f"Keywords: {keywords}"
    )

    return {
        "final_result": final_result
    }


# 6. Build the graph
graph_builder = StateGraph(WorkflowState)

graph_builder.add_node("identify_topic", identify_topic)
graph_builder.add_node("analyze_intent", analyze_intent)
graph_builder.add_node("extract_keywords", extract_keywords)
graph_builder.add_node("combine_results", combine_results)

graph_builder.add_edge(START, "identify_topic")

# Fan-out: execute independent nodes
graph_builder.add_edge("identify_topic", "analyze_intent")
graph_builder.add_edge("identify_topic", "extract_keywords")

# Fan-in: wait for both branches
graph_builder.add_edge("analyze_intent", "combine_results")
graph_builder.add_edge("extract_keywords", "combine_results")

graph_builder.add_edge("combine_results", END)

graph = graph_builder.compile()


# 7. Run the graph
input_data = {
    "question": " What is LangGraph used for in AI?",
    "topic": "",
    "intent": "",
    "keywords": [],
    "final_result": ""
}

result = graph.invoke(input_data)

print("\nFinal Workflow Result:\n")
print("Question:", result["question"])
print(result["final_result"])