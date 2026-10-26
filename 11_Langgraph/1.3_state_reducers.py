from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END


# 1. Define the shared state
class WorkflowState(TypedDict):
    question: str
    messages: Annotated[list[str], operator.add]


# 2. First node
def receive_question(state: WorkflowState):
    question = state["question"]

    return {"messages": [f"Received question: {question}"]}


# 3. Second node
def analyze_question(state: WorkflowState):
    question = state["question"]

    return {"messages": [f"Analyzing question: {question}"]}


# 4. Third node
def generate_answer(state: WorkflowState):
    question = state["question"]

    answer = f"You asked: {question}"

    return {"messages": [f"Generated answer: {answer}"]}


def validate_answer(state: WorkflowState):
    return {"messages": ["Answer validation completed."]}


# 5. Create the graph
graph_builder = StateGraph(WorkflowState)


# 6. Add nodes
graph_builder.add_node("receive_question", receive_question)
graph_builder.add_node("analyze_question", analyze_question)
graph_builder.add_node("generate_answer", generate_answer)
graph_builder.add_node("validate_answer", validate_answer)


# 7. Define workflow
graph_builder.add_edge(START, "receive_question")
graph_builder.add_edge("receive_question", "analyze_question")
graph_builder.add_edge("analyze_question", "generate_answer")
graph_builder.add_edge("generate_answer", "validate_answer")
graph_builder.add_edge("validate_answer", END)


# 8. Compile graph
graph = graph_builder.compile()


# 9. Execute graph
result = graph.invoke({"question": "What is LangGraph?", "messages": []})


# 10. Display result
print("\nFinal result:")
print(result)

print("\nMessages:")
for message in result["messages"]:
    print(message)
