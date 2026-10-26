
import json
from typing import Annotated, TypedDict

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
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
                "context": "No relevant handbook information was found.",
                "sources": [],
            },
            ensure_ascii=False,
        )

    sources = [
        {
            "chunk_index": result["chunk_index"],
            "score": result["score"],
            "excerpt": result["chunk"][:500],
        }
        for result in results
    ]

    return json.dumps(
        {
            "context": context,
            "sources": sources,
        },
        ensure_ascii=False,
    )


# --------------------------------------------------
# 4. Configure LLMs
# --------------------------------------------------

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)

# IMPORTANT:
# This LLM has NO tools bound to it.
# General-knowledge questions are answered directly.
general_llm = llm

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

    return any(
        keyword in normalized_question
        for keyword in HANDBOOK_KEYWORDS
    )


# --------------------------------------------------
# 6. System prompts
# --------------------------------------------------

GENERAL_SYSTEM_PROMPT = """
You are a helpful general-purpose AI assistant.

Answer general-knowledge questions directly using your
existing knowledge.

Do not call tools or return tool-call JSON.
For lists, provide a clear, organized answer.
For calculations, answer simple arithmetic directly.
If you are uncertain, explain the uncertainty rather
than inventing facts.
"""

HANDBOOK_SYSTEM_PROMPT = """
You are an employee handbook assistant.

Use the search_handbook tool to answer questions about
company policies, leave, benefits, and reimbursements.

Base handbook answers on the retrieved handbook context.
Do not invent company policies or details.

If the handbook does not contain relevant information,
clearly tell the user that the information was not found.
"""


# --------------------------------------------------
# 7. Chatbot node
# --------------------------------------------------

def chatbot(state: AgentState):
    latest_question = ""

    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            latest_question = str(message.content)
            break

    if is_handbook_question(latest_question):
        selected_llm = llm_with_handbook_tools
        system_prompt = HANDBOOK_SYSTEM_PROMPT

    elif is_calculation_question(latest_question):
        selected_llm = llm_with_calculation_tools
        system_prompt = GENERAL_SYSTEM_PROMPT

    else:
        # General knowledge: plain LLM, no tools.
        selected_llm = general_llm
        system_prompt = GENERAL_SYSTEM_PROMPT

    response = selected_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            *state["messages"],
        ]
    )

    return {"messages": [response]}


def is_calculation_question(question: str) -> bool:
    """Identify simple arithmetic requests."""

    normalized = question.casefold()

    calculation_terms = (
        "calculate",
        "add ",
        "multiply",
        "product of",
        "sum of",
    )

    return any(
        term in normalized
        for term in calculation_terms
    ) or (
        "+" in question
        or "*" in question
    )


# --------------------------------------------------
# 8. Build graph
# --------------------------------------------------

graph_builder = StateGraph(AgentState)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_node(
    "calculation_tools",
    ToolNode(calculation_tools),
)

graph_builder.add_node(
    "handbook_tools",
    ToolNode(handbook_tools),
)

graph_builder.add_edge(START, "chatbot")


def route_after_chatbot(state: AgentState):
    """Route only valid tool calls to the correct tool node."""

    last_message = state["messages"][-1]

    tool_calls = getattr(last_message, "tool_calls", None)

    if not tool_calls:
        return END

    latest_question = ""

    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            latest_question = str(message.content)
            break

    if is_handbook_question(latest_question):
        allowed_tools = {
            "add_numbers",
            "multiply_numbers",
            "search_handbook",
        }
        destination = "handbook_tools"

    elif is_calculation_question(latest_question):
        allowed_tools = {
            "add_numbers",
            "multiply_numbers",
        }
        destination = "calculation_tools"

    else:
        # A general-knowledge request must not execute tools.
        return "invalid_tool_call"

    for tool_call in tool_calls:
        if tool_call["name"] not in allowed_tools:
            return "invalid_tool_call"

    return destination


def reject_invalid_tool_call(state: AgentState):
    """Return a safe text response if an unexpected tool call occurs."""

    return {
        "messages": [
            # The unexpected tool call is not shown as the answer.
            # A new human-readable response replaces it.
            # The graph's state still retains its earlier history.
            SystemMessage(
                content=(
                    "The requested operation was not available. "
                    "Please answer the user's question directly "
                    "without calling tools."
                )
            )
        ]
    }


graph_builder.add_node(
    "invalid_tool_call",
    reject_invalid_tool_call,
)

graph_builder.add_conditional_edges(
    "chatbot",
    route_after_chatbot,
)

graph_builder.add_edge("calculation_tools", "chatbot")
graph_builder.add_edge("handbook_tools", "chatbot")
graph_builder.add_edge("invalid_tool_call", "chatbot")


# --------------------------------------------------
# 9. Persistent SQLite checkpoint
# --------------------------------------------------

checkpointer_context = SqliteSaver.from_conn_string(
    "langgraph_fastapi_memory.sqlite"
)

checkpointer = checkpointer_context.__enter__()

graph = graph_builder.compile(
    checkpointer=checkpointer
)
