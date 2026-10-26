import json
from typing import TypedDict, Annotated

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
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

from rag.rag_pipeline import (
    initialize_vector_store,
    retrieve_context,
)


# --------------------------------------------------
# 1. Define graph state
# --------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# --------------------------------------------------
# 2. Initialize handbook vector store
# --------------------------------------------------

vector_index, document_chunks = initialize_vector_store()


# --------------------------------------------------
# 3. Define tools
# --------------------------------------------------

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b


@tool
def search_handbook(question: str) -> str:
    """
    Search the employee handbook for company policies,
    leave, benefits, reimbursement, onboarding,
    and company procedures.
    """

    context, results = retrieve_context(
        question=question,
        index=vector_index,
        chunks=document_chunks,
        top_k=3,
    )

    if not results:
        return json.dumps(
            {
                "context": (
                    "No relevant handbook information was found."
                ),
                "sources": [],
            },
            ensure_ascii=False,
        )

    sources = []

    for result in results:
        sources.append(
            {
                "chunk_index": result["chunk_index"],
                "score": result["score"],
                "excerpt": result["chunk"][:500],
            }
        )

    return json.dumps(
        {
            "context": context,
            "sources": sources,
        },
        ensure_ascii=False,
    )


# --------------------------------------------------
# 4. Configure LLM and tool bindings
# --------------------------------------------------

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)

calculation_tools = [
    add_numbers,
    multiply_numbers,
]

handbook_tools = [
    add_numbers,
    multiply_numbers,
    search_handbook,
]

llm_with_calculation_tools = llm.bind_tools(
    calculation_tools
)

llm_with_handbook_tools = llm.bind_tools(
    handbook_tools
)


# --------------------------------------------------
# 5. Detect handbook-related questions
# --------------------------------------------------

HANDBOOK_KEYWORDS = (
    "handbook",
    "company policy",
    "company policies",
    "employee policy",
    "employee policies",
    "annual leave",
    "sick leave",
    "leave balance",
    "reimbursement",
    "internet expenses",
    "expense claim",
    "expense claims",
    "work from home",
    "working hours",
    "holiday calendar",
    "onboarding",
    "company benefits",
    "employee benefits",
    "company holidays",
    "travel reimbursement",
    "transport reimbursement",
    "parental leave",
    "maternity leave",
    "paternity leave",
    "notice period",
    "salary policy",
    "expense policy",
    "company procedure",
    "company procedures",
)


def is_handbook_question(question: str) -> bool:
    """Identify questions about company handbook information."""

    normalized_question = question.casefold()

    for keyword in HANDBOOK_KEYWORDS:
        if keyword in normalized_question:
            return True

    return False


# --------------------------------------------------
# 6. Chatbot node
# --------------------------------------------------
def chatbot(state: AgentState):
    latest_question = ""

    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            latest_question = str(message.content)
            break

    if is_handbook_question(latest_question):
        selected_llm = llm_with_handbook_tools
        system_prompt = (
            "Answer questions about company policies using the "
            "handbook search tool. If the handbook does not contain "
            "the answer, clearly say so."
        )
    else:
        selected_llm = llm_with_calculation_tools
        system_prompt = (
            "You are a helpful general-purpose assistant. "
            "Answer general-knowledge questions directly using your "
            "existing knowledge. Do not restrict answers to a handbook. "
            "The capital of Iceland is Reykjavík. "
            "Do not call handbook search for general-knowledge questions."
        )

    response = selected_llm.invoke(
        [SystemMessage(content=system_prompt)] + state["messages"]
    )

    return {"messages": [response]}
# --------------------------------------------------
# 7. Build graph with separate tool-routing paths
# --------------------------------------------------

graph_builder = StateGraph(AgentState)

graph_builder.add_node(
    "chatbot",
    chatbot,
)

graph_builder.add_node(
    "calculation_tools",
    ToolNode(calculation_tools),
)

graph_builder.add_node(
    "handbook_tools",
    ToolNode(handbook_tools),
)

graph_builder.add_edge(
    START,
    "chatbot",
)


def route_after_chatbot(state: AgentState):
    """
    Route tool calls only to tools available for the
    latest question.
    """

    messages = state["messages"]

    last_message = messages[-1]

    if not getattr(last_message, "tool_calls", None):
        return END

    latest_question = ""

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            latest_question = str(message.content)
            break

    allowed_tool_names = (
        {"add_numbers", "multiply_numbers", "search_handbook"}
        if is_handbook_question(latest_question)
        else {"add_numbers", "multiply_numbers"}
    )

    # Validate requested tools before executing them.
    for tool_call in last_message.tool_calls:
        if tool_call["name"] not in allowed_tool_names:
            return END

    if is_handbook_question(latest_question):
        return "handbook_tools"

    return "calculation_tools"


graph_builder.add_conditional_edges(
    "chatbot",
    route_after_chatbot,
)

graph_builder.add_edge(
    "calculation_tools",
    "chatbot",
)

graph_builder.add_edge(
    "handbook_tools",
    "chatbot",
)


# --------------------------------------------------
# 8. Persistent SQLite checkpoint
# --------------------------------------------------

checkpointer_context = SqliteSaver.from_conn_string(
    "langgraph_fastapi_memory.sqlite"
)

checkpointer = checkpointer_context.__enter__()

graph = graph_builder.compile(
    checkpointer=checkpointer
)
