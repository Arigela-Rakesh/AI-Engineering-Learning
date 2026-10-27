import json

from contextlib import asynccontextmanager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import graph_builder
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, ToolMessage


from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncSqliteSaver.from_conn_string(
        "langgraph_fastapi_memory.sqlite"
    ) as checkpointer:
        app.state.graph = graph_builder.compile(
            checkpointer=checkpointer
        )
        yield


# 1. Create FastAPI application
app = FastAPI(
    title="LangGraph AI Assistant",
    description="FastAPI integration with LangGraph",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Define request schema
class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        description="User's message",
    )

    thread_id: str = Field(
        default="conversation-1",
        description="Conversation thread identifier",
    )


# 3. Define response schema
class HandbookSource(BaseModel):
    chunk_index: int
    score: float
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    sources: list[HandbookSource] = Field(
        default_factory=list,
        description="Handbook evidence retrieved for the answer",
    )
    
def extract_sources(messages):
    """
    Extract handbook source metadata from the latest user turn.
    """

    sources = []

    # Find the latest user message.
    latest_human_index = -1

    for index in range(len(messages) - 1, -1, -1):

        if isinstance(messages[index], HumanMessage):
            latest_human_index = index
            break

    if latest_human_index == -1:
        return sources

    # Inspect messages after the latest user message.
    for message in messages[latest_human_index + 1:]:

        if not isinstance(message, ToolMessage):
            continue

        if message.name != "search_handbook":
            continue

        try:
            tool_data = json.loads(message.content)

            tool_sources = tool_data.get("sources", [])

            sources.extend(tool_sources)

        except (json.JSONDecodeError, AttributeError, TypeError):
            continue

    return sources

# 4. Create chat endpoint
# 4. Create chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: Request,
    chat_request: ChatRequest,
) -> ChatResponse:

    config = {
        "configurable": {
            "thread_id": chat_request.thread_id
        }
    }

    # Access the graph compiled during FastAPI startup.
    request_graph = request.app.state.graph

    result = await request_graph.ainvoke(
        {
            "messages": [
                HumanMessage(content=chat_request.message)
            ]
        },
        config=config,
    )

    messages = result["messages"]

    # Get the final AI response.
    answer = messages[-1].content

    # Extract handbook sources from the latest user turn.
    sources = extract_sources(messages)

    return {
        "answer": answer,
        "thread_id": chat_request.thread_id,
        "sources": sources,
    }