from typing import TypedDict, Annotated

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
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


# 2. Create tools
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b


# 3. Register tools
tools = [
    add_numbers,
    multiply_numbers,
]


# 4. Create the local LLM
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
        "You are a helpful and concise AI assistant. "
        "Answer personal questions directly using conversation history. "
        "If the user asks their name, return the name they previously provided. "
        "Never use a calculation tool for names, greetings, or personal questions. "
        "Never invent a calculation when the user has not asked for one. "
        "Use add_numbers only when the user explicitly asks for addition. "
        "Use multiply_numbers only when the user explicitly asks for multiplication. "
        "If the user provides a single number without a calculation request, "
        "respond conversationally without calling a tool. "
        "When a tool is needed, use the correct tool and its actual result. "
        "Keep responses short and natural."
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


# 8. Define graph edges
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


# 9. Create SQLite checkpoint storage
with SqliteSaver.from_conn_string(
    "langgraph_capstone_memory.sqlite"
) as checkpointer:

    graph = graph_builder.compile(
        checkpointer=checkpointer
    )

    # 10. Configure the conversation
    config = {
        "configurable": {
            "thread_id": "capstone-conversation-1"
        }
    }

    print("=" * 50)
    print("LangGraph Capstone AI Assistant")
    print("=" * 50)

    print("\nCommands:")
    print("  exit  - End the conversation")
    print("  reset - Start a new conversation thread")
    print()

    # 11. Chat loop
    while True:

        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("\nConversation ended.")
            break

        # Start a separate thread when requested
        if user_input.lower() == "reset":
            config = {
                "configurable": {
                    "thread_id": "capstone-conversation-2"
                }
            }

            print("\nStarted a new conversation thread.\n")
            continue

        # 12. Invoke the graph
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

        # 13. Print the final answer
        final_message = result["messages"][-1]

        if isinstance(final_message, AIMessage):
            print("\nAssistant:")
            print(final_message.content)
            print()