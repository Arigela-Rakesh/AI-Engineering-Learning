from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# 1. Define the shared state
class WorkflowState(TypedDict):
    question: str
    topic: str
    answer: str
    validation:str


# 2. Define nodes
def identify_topic(state: WorkflowState):
    question = state["question"].lower()

    if "python" in question:
        topic = "Python"
    elif "langgraph" in question:
        topic = "LangGraph"
    else:
        topic = "General"

    return {"topic": topic}


def generate_answer(state: WorkflowState):
    topic = state["topic"]

    if topic == "Python":
        answer = "Python is widely used in AI and machine learning."
    elif topic == "LangGraph":
        answer = "LangGraph is used to build stateful AI workflows."
    else:
        answer = "This is a general question."

    return {"answer": answer}

def validate_answer(state: WorkflowState):
    answer=state["answer"]
    
    if(len(answer)==0):
        validation="Empty answer"
    else:
        validation= "Valid answer"
    
    return {
        "validation": validation
    }

# 3. Build the graph
graph_builder = StateGraph(WorkflowState)

graph_builder.add_node("identify_topic", identify_topic)
graph_builder.add_node("generate_answer", generate_answer)
graph_builder.add_node("validate_answer", validate_answer)

graph_builder.add_edge(START, "identify_topic")
graph_builder.add_edge("identify_topic", "generate_answer")
graph_builder.add_edge("generate_answer", "validate_answer")
graph_builder.add_edge("validate_answer", END)

graph = graph_builder.compile()


# 4. Visualize the graph
graph_structure = graph.get_graph()

print("\nGraph Nodes:\n")
print(graph_structure.nodes)

print("\nGraph Edges:\n")
print(graph_structure.edges)

print("\nMermaid Diagram:\n")
print(graph_structure.draw_mermaid())


# 5. Run the graph
input_data = {"question": "What is LangGraph used for?", "topic": "", "answer": "","validation":""}

result = graph.invoke(input_data)

print("\nFinal Result:\n")
print(result)
