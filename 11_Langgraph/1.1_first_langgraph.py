from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# 1. Define the shared state
class WorkflowState(TypedDict):
    question: str
    topic: str
    answer: str


# 2. Define the first node
def receive_question(state: WorkflowState):
    question = state["question"]

    print("Received question:", question)

    return {"question": question}


# 3. Define the new identify_topic node
def identify_topic(state: WorkflowState):
    return {"topic": "AI Engineering"}


# 4. Define the generate_answer node
def generate_answer(state: WorkflowState):
    question = state["question"]
    topic = state["topic"]

    answer = f"You asked: {question}\nTopic: {topic}"

    return {"answer": answer}


# 5. Create the graph
graph_builder = StateGraph(WorkflowState)


# 6. Add nodes
graph_builder.add_node("receive_question", receive_question)
graph_builder.add_node("identify_topic", identify_topic)
graph_builder.add_node("generate_answer", generate_answer)


# 7. Define the workflow
graph_builder.add_edge(START, "receive_question")
graph_builder.add_edge("receive_question", "identify_topic")
graph_builder.add_edge("identify_topic", "generate_answer")
graph_builder.add_edge("generate_answer", END)


# 8. Compile the graph
graph = graph_builder.compile()


# 9. Execute the graph
result = graph.invoke({"question": "What is LangGraph?", "topic": "", "answer": ""})


# 10. Display the result
print("\nFinal result:")
print(result)
