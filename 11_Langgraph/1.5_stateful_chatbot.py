from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver


# 1. Define the shared state
class WorkflowState(TypedDict):
    messages: Annotated[list, add_messages]


# 2. Initialize the local chat model
chat_model = ChatOllama(model="llama3.2:3b", temperature=0)


# 3. Define the chatbot node
def chatbot(state: WorkflowState):
    messages = state["messages"]

    response = chat_model.invoke(messages)

    return {"messages": [response]}


# 4. Create the graph
graph_builder = StateGraph(WorkflowState)


# 5. Add the chatbot node
graph_builder.add_node("chatbot", chatbot)


# 6. Define the workflow
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)


# 7. Create the checkpointer
memory = InMemorySaver()


# 8. Compile the graph
graph = graph_builder.compile(checkpointer=memory)


# 9. Define the conversation thread
config = {"configurable": {"thread_id": "chatbot-session-1"}}


# 10. First user message
result_1 = graph.invoke(
    {"messages": [HumanMessage(content="My favorite programming language is Python.")]}, config=config
)

print("First response:")
print(result_1["messages"][-1].content)


# 11. Second user message
result_2 = graph.invoke(
    {"messages": [HumanMessage(content="What is my favorite programming language?")]}, config=config
)

print("\nSecond response:")
print(result_2["messages"][-1].content)
