from typing import TypedDict, Annotated

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
)

from langchain_core.tools import tool

from langchain_ollama import ChatOllama

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.graph.message import add_messages

from langgraph.prebuilt import ToolNode, tools_condition


# 1. Define the shared state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# 2. Create a calculator tool
@tool
def multiply_numbers(a: int, b: int) -> int:
    """
    Multiply two numbers and return the result.
    """
    return a * b

@tool
def add_numbers(a: int, b: int) -> int:
    """
        Add two numbers and return the result.
    """
    return a + b

# 3. Register the tools
tools = [multiply_numbers,add_numbers]


# 4. Create the LLM
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)


# 5. Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)


# 6. Create the chatbot node
def chatbot(state: AgentState):
    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# 7. Build the graph
graph_builder = StateGraph(AgentState)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_node(
    "tools",
    ToolNode(tools)
)


# 8. Define the graph edges
graph_builder.add_edge(START, "chatbot")

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge(
    "tools",
    "chatbot",
)

graph_builder.add_edge(
    "chatbot",
    END,
)


# 9. Compile the graph
graph = graph_builder.compile()


# 10. Run the agent
question = "What is 245 adding by 12?"

result = graph.invoke(
    {
        "messages": [
            HumanMessage(content=question)
        ]
    }
)


# 11. Print the conversation
print("\nAgent Conversation:\n")

for message in result["messages"]:

    if isinstance(message, HumanMessage):
        print("User:")
        print(message.content)

    elif isinstance(message, AIMessage):

        if message.tool_calls:
            print("\nAI Tool Call:")
            print(message.tool_calls)

        elif message.content:
            print("\nAI Answer:")
            print(message.content)

    elif isinstance(message, ToolMessage):
        print("\nTool Result:")
        print(message.content)