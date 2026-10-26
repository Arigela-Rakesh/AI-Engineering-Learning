from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


# 1. Define the shared state
class WorkflowState(TypedDict):
    question: str
    topic: str
    answer: str


# 2. Identify the topic
def identify_topic(state: WorkflowState):
    question = state["question"].lower()

    if "python" in question:
        topic = "Python"
    elif "langgraph" in question:
        topic = "LangGraph"
    else:
        topic = "General"

    return {"topic": topic}


# 3. Generate an answer
def generate_answer(state: WorkflowState):
    topic = state["topic"]

    if topic == "Python":
        answer = "Python is widely used in AI and machine learning."
    elif topic == "LangGraph":
        answer = "LangGraph is used to build stateful AI workflows."
    else:
        answer = "This is a general question."

    return {"answer": answer}


# 4. Build the graph
graph_builder = StateGraph(WorkflowState)

graph_builder.add_node("identify_topic", identify_topic)
graph_builder.add_node("generate_answer", generate_answer)

graph_builder.add_edge(START, "identify_topic")
graph_builder.add_edge("identify_topic", "generate_answer")
graph_builder.add_edge("generate_answer", END)


# 5. Compile with a checkpointer
checkpointer = InMemorySaver()

graph = graph_builder.compile(
    checkpointer=checkpointer
)


# 6. Configure the workflow
config = {
    "configurable": {
        "thread_id": "state-inspection-2"
    }
}


# 7. Invoke the workflow
input_data = {
    "question": "How is Python used in AI?",
    "topic": "",
    "answer": ""
}

result = graph.invoke(input_data, config=config)

print("\nFinal Workflow Result:\n")
print(result)


# 8. Inspect the latest checkpoint
snapshot = graph.get_state(config)

print("\nLatest Checkpoint Values:\n")
print(snapshot.values)

print("\nNext Node:\n")
print(snapshot.next)


# 9. Inspect checkpoint history
print("\nCheckpoint History:\n")

for history_snapshot in graph.get_state_history(config):
    print("State:")
    print(history_snapshot.values)

    print("Next:")
    print(history_snapshot.next)

    print("-" * 50)