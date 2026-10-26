from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages

from langchain_core.messages import HumanMessage, AIMessage


# 1. Define the shared state
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


# 2. Create a chatbot node
def chatbot(state: ChatState):
    messages = state["messages"]

    last_message = messages[-1].content

    if "my name is" in last_message.lower():
        name = last_message.split("is")[-1].strip()

        answer = f"Nice to meet you, {name}!"

    elif "what is my name" in last_message.lower():
        previous_name = "I need to check the conversation history."

        for message in messages:
            if isinstance(message, HumanMessage):
                content = message.content.lower()

                if "my name is" in content:
                    previous_name = message.content.split("is")[-1].strip()

        answer = f"Your name is {previous_name}."

    else:
        answer = "I am a simple persistent chatbot."

    return {
        "messages": [
            AIMessage(content=answer)
        ]
    }


# 3. Build the graph
graph_builder = StateGraph(ChatState)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)


# 4. Create SQLite checkpoint storage
with SqliteSaver.from_conn_string(
    "langgraph_chat_memory.sqlite"
) as checkpointer:

    graph = graph_builder.compile(
        checkpointer=checkpointer
    )

    # 5. Use a persistent thread ID
    config = {
        "configurable": {
            "thread_id": "chat-memory-2"
        }
    }

    # 6. Get user input
    user_input = input("You: ")

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ]
        },
        config=config
    )

    # 7. Print the chatbot response
    print("\nAssistant:")

    print(
        result["messages"][-1].content
    )