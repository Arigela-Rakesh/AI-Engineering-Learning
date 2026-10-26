from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


# 1. Define the shared state
class WorkflowState(TypedDict):
    messages: Annotated[list[str], operator.add]


# 2. Define the node
def chatbot(state: WorkflowState):
    messages = state["messages"]

    latest_message = messages[-1]

    response = f"I am an AI assistant. You said: {latest_message}"

    return {"messages": [response]}


# 3. Create the graph
graph_builder = StateGraph(WorkflowState)


# 4. Add the node
graph_builder.add_node("chatbot", chatbot)


# 5. Define the workflow
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)


# 6. Create an in-memory checkpointer
memory = InMemorySaver()


# 7. Compile the graph with the checkpointer
graph = graph_builder.compile(checkpointer=memory)


# 8. Define a conversation thread
config = {"configurable": {"thread_id": "conversation-1"}}


# 9. First invocation
result_1 = graph.invoke({"messages": ["Hello"]}, config=config)

print("First invocation:")
print(result_1)


# 10. Second invocation using the same thread
result_2 = graph.invoke({"messages": ["What is LangGraph?"]}, config=config)

print("\nSecond invocation:")
print(result_2)


result_3 = graph.invoke({"messages": ["Thank you"]}, config=config)

print("\nThird invocation:")
print(result_3)
