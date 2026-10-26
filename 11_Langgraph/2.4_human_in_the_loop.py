from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


# 1. Define the shared state
class WorkflowState(TypedDict):
    question: str
    answer: str
    approved: bool
    status: str


# 2. Generate an answer
def generate_answer(state: WorkflowState):
    question = state["question"]

    answer = (
        f"AI-generated answer for your question: {question}"
    )

    return {"answer": answer}


# 3. Pause and ask for human approval
def human_review(state: WorkflowState):
    decision = interrupt(
        {
            "message": "Please review the generated answer.",
            "question": state["question"],
            "answer": state["answer"],
            "instruction": "Resume with True to approve or False to reject."
        }
    )

    return {"approved": decision}


# 4. Mark the answer as approved
def mark_approved(state: WorkflowState):
    return {"status": "Approved by human."}


# 5. Mark the answer as rejected
def mark_rejected(state: WorkflowState):
    return {"status": "Rejected by human."}


# 6. Define conditional routing
def route_after_review(state: WorkflowState):
    if state["approved"]:
        return "mark_approved"

    return "mark_rejected"


# 7. Build the graph
graph_builder = StateGraph(WorkflowState)

graph_builder.add_node("generate_answer", generate_answer)
graph_builder.add_node("human_review", human_review)
graph_builder.add_node("mark_approved", mark_approved)
graph_builder.add_node("mark_rejected", mark_rejected)

graph_builder.add_edge(START, "generate_answer")
graph_builder.add_edge("generate_answer", "human_review")

graph_builder.add_conditional_edges(
    "human_review",
    route_after_review,
    {
        "mark_approved": "mark_approved",
        "mark_rejected": "mark_rejected"
    }
)

graph_builder.add_edge("mark_approved", END)
graph_builder.add_edge("mark_rejected", END)


# 8. Compile with a checkpointer
checkpointer = InMemorySaver()

graph = graph_builder.compile(
    checkpointer=checkpointer
)


# 9. Configure the workflow thread
config = {
    "configurable": {
        "thread_id": "human-review-2"
    }
}


# 10. Start the workflow
input_data = {
    "question": "How can LangGraph help build AI agents?",
    "answer": "",
    "approved": False,
    "status": ""
}

result = graph.invoke(input_data, config=config)

print("\nWorkflow Paused for Human Review:\n")
print(result)


# 11. Ask the human for a decision
decision = input(
    "\nDo you approve this answer? (yes/no): "
).strip().lower()

if decision == "yes":
    approval_value = True
else:
    approval_value = False


# 12. Resume the workflow
final_result = graph.invoke(
    Command(resume=approval_value),
    config=config
)

print("\nFinal Workflow Result:\n")
print(final_result)