import json
import sys
from pathlib import Path
from typing import Annotated, TypedDict


from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


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
# 3. Define handbook RAG tool
# --------------------------------------------------


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
                "context": ("No relevant handbook information was found."),
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
# 4. Configure Ollama
# --------------------------------------------------

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)

# General-knowledge questions use this LLM directly.
# No tools are bound to this instance.
general_llm = llm


# --------------------------------------------------
# 5. Configure the MCP calculator client
# --------------------------------------------------

MCP_SERVER_PATH = Path(__file__).parent / "mcp_server" / "server.py"

if not MCP_SERVER_PATH.exists():
    raise FileNotFoundError(f"MCP server file not found: {MCP_SERVER_PATH}")

mcp_client = MultiServerMCPClient(
    {
        "calculator": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(MCP_SERVER_PATH)],
        }
    }
)


# --------------------------------------------------
# 6. Asynchronous MCP calculator tool
# --------------------------------------------------


@tool
async def calculate_via_mcp(
    a: float,
    b: float,
    operation: str,
) -> str:
    """
    Perform arithmetic through the MCP calculator server.

    Supported operations: add, subtract, multiply, divide.
    """

    if operation not in {
        "add",
        "subtract",
        "multiply",
        "divide",
    }:
        raise ValueError(
            "Unsupported operation. Use add, subtract, " "multiply, or divide."
        )

    if operation == "divide" and b == 0:
        raise ValueError("Cannot divide by zero.")

    # Discover the MCP tools asynchronously.
    discovered_tools = await mcp_client.get_tools()

    calculate_tool = None

    for discovered_tool in discovered_tools:
        if discovered_tool.name == "calculate":
            calculate_tool = discovered_tool
            break

    if calculate_tool is None:
        raise RuntimeError(
            "The MCP calculator server did not expose " "a calculate tool."
        )

    # Invoke the MCP tool asynchronously.
    result = await calculate_tool.ainvoke(
        {
            "a": a,
            "b": b,
            "operation": operation,
        }
    )

    # Convert MCP content blocks into readable text.
    if isinstance(result, list):
        text_parts = []

        for item in result:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(str(item["text"]))

        if text_parts:
            return "\n".join(text_parts)

    return str(result)


# --------------------------------------------------
# 7. Configure tool-enabled LLMs
# --------------------------------------------------

calculation_tools = [
    calculate_via_mcp,
]

handbook_tools = [
    calculate_via_mcp,
    search_handbook,
]

llm_with_calculation_tools = llm.bind_tools(calculation_tools)

llm_with_handbook_tools = llm.bind_tools(handbook_tools)


# --------------------------------------------------
# 8. Detect handbook-related questions
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
# 9. Detect calculation questions
# --------------------------------------------------


def is_calculation_question(question: str) -> bool:
    """Identify common arithmetic requests."""

    normalized = question.casefold()

    calculation_terms = (
        "calculate",
        "add ",
        "multiply",
        "multiplied",
        "subtract",
        "divided",
        "divide",
        "product of",
        "sum of",
    )

    for term in calculation_terms:
        if term in normalized:
            return True

    return "+" in question or "*" in question or "/" in question


# --------------------------------------------------
# 10. System prompts
# --------------------------------------------------

GENERAL_SYSTEM_PROMPT = """
You are a helpful general-purpose AI assistant.

Answer general-knowledge questions directly using your
existing knowledge.

Do not call tools or return tool-call JSON.
For lists, provide a clear, organized answer.
For simple arithmetic, answer directly if no calculation
tool was requested by the application.
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

When relevant handbook context is retrieved, answer using
that context and preserve the source information returned
by the tool.
"""

CALCULATION_SYSTEM_PROMPT = """
You are a helpful assistant with access to an MCP calculator.

Use calculate_via_mcp for arithmetic requests.
Choose the appropriate operation: add, subtract,
multiply, or divide.

Do not invent calculator results.
After the tool returns its result, explain the answer
clearly to the user.
"""


# --------------------------------------------------
# 11. Asynchronous chatbot node
# --------------------------------------------------


async def chatbot(state: AgentState):
    """Select the correct LLM configuration for each question."""

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
        system_prompt = CALCULATION_SYSTEM_PROMPT

    else:
        selected_llm = general_llm
        system_prompt = GENERAL_SYSTEM_PROMPT

    response = await selected_llm.ainvoke(
        [
            SystemMessage(content=system_prompt),
            *state["messages"],
        ]
    )

    return {"messages": [response]}


# --------------------------------------------------
# 12. Build LangGraph
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
    """Route only expected tool calls."""

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
            "calculate_via_mcp",
            "search_handbook",
        }
        destination = "handbook_tools"

    elif is_calculation_question(latest_question):
        allowed_tools = {
            "calculate_via_mcp",
        }
        destination = "calculation_tools"

    else:
        # General questions should never request tools because
        # the general-purpose LLM has no tools bound to it.
        return END

    for tool_call in tool_calls:
        if tool_call["name"] not in allowed_tools:
            return END

    return destination


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
# 13. Expose graph builder
# --------------------------------------------------

# The graph is compiled during FastAPI startup,
# after the asynchronous SQLite checkpointer is opened.
