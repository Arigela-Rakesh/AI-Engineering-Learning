from typing import TypedDict, Annotated
from langchain_core.messages import SystemMessage
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
from langgraph.checkpoint.sqlite import SqliteSaver


# 1. Define the shared state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# 2. Create calculator tools
@tool
def multiply_numbers(a: int, b: int) -> int:
    """
    Multiply two numbers.
    """
    return a * b


@tool
def add_numbers(a: int, b: int) -> int:
    """
    Add two numbers.
    """
    return a + b


# 3. Register the tools
tools = [
    multiply_numbers,
    add_numbers,
]


# 4. Create the LLM
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)


# 5. Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)


# 6. Create the chatbot node
def chatbot(state: AgentState):
    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. "
            "Use the conversation history to answer questions "
            "about the user's name and other personal details "
            "they have shared. "
            "Do not use calculator tools for personal questions "
            "or general conversation. "
            "Use multiply_numbers only for multiplication. "
            "Use add_numbers only for addition. "
            "If the answer is already available in the conversation "
            "history, answer directly without calling a tool."
        )
    )

    messages = [
        system_message
    ] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


# 7. Build the graph
graph_builder = StateGraph(AgentState)

graph_builder.add_node(
    "chatbot",
    chatbot
)

graph_builder.add_node(
    "tools",
    ToolNode(tools)
)


# 8. Define the graph edges
graph_builder.add_edge(
    START,
    "chatbot"
)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge(
    "tools",
    "chatbot"
)

graph_builder.add_edge(
    "chatbot",
    END
)


# 9. Create persistent checkpoint storage
with SqliteSaver.from_conn_string(
    "stateful_agent_memory.sqlite"
) as checkpointer:

    graph = graph_builder.compile(
        checkpointer=checkpointer
    )

    # 10. Configure the conversation thread
    config = {
        "configurable": {
            "thread_id": "agent-conversation-1"
        }
    }

    print("Stateful AI Agent")
    print("Type 'exit' to stop.\n")

    # 11. Chat loop
    while True:

        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Chat ended.")
            break

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=user_input
                    )
                ]
            },
            config=config
        )

        # 12. Print the final response
        final_message = result["messages"][-1]

        print("\nAssistant:")
        print(final_message.content)
        print()