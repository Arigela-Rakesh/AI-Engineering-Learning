
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# --------------------------------------------------
# 1. Load the saved FAISS vector store
# --------------------------------------------------

base_directory = Path(__file__).parent

vector_store_path = (
    base_directory
    / "faiss_company_policy"
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.load_local(
    str(vector_store_path),
    embedding_model,
    allow_dangerous_deserialization=True,
)


# --------------------------------------------------
# 2. Create the retriever
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# --------------------------------------------------
# 3. Create the local LLM
# --------------------------------------------------

chat_model = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)


# --------------------------------------------------
# 4. Create the question rewriting prompt
# --------------------------------------------------

contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Given the conversation history and the latest question,
rewrite the latest question as a standalone question.

Do not answer the question.
If the latest question is already standalone,
return it unchanged.
"""
        ),
        (
            "placeholder",
            "{chat_history}",
        ),
        (
            "human",
            "{question}",
        ),
    ]
)


# --------------------------------------------------
# 5. Create the question rewriting chain
# --------------------------------------------------

contextualize_chain = (
    contextualize_prompt
    | chat_model
    | StrOutputParser()
)


# --------------------------------------------------
# 6. Create the answer prompt
# --------------------------------------------------

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful company policy assistant.

Answer using only the provided company policy context.

If the context does not contain the answer,
say that the company policy document does not
provide that information.

Do not invent facts.

Context:
{context}
"""
        ),
        (
            "placeholder",
            "{chat_history}",
        ),
        (
            "human",
            "{question}",
        ),
    ]
)


# --------------------------------------------------
# 7. Format retrieved documents
# --------------------------------------------------

def format_documents(documents):
    formatted_documents = []

    for document in documents:
        formatted_documents.append(
            document.page_content
        )

    return "\n\n".join(formatted_documents)


# --------------------------------------------------
# 8. Build the conversational RAG chain
# --------------------------------------------------

def retrieve_context(inputs):
    question = inputs["question"]
    chat_history = inputs["chat_history"]

    # Rewrite the question when conversation history exists.
    if len(chat_history) > 0:
        standalone_question = contextualize_chain.invoke(
            {
                "question": question,
                "chat_history": chat_history,
            }
        )
    else:
        standalone_question = question

    # Retrieve documents using the standalone question.
    retrieved_documents = retriever.invoke(
        standalone_question
    )

    return format_documents(retrieved_documents)


rag_chain = (
    RunnablePassthrough.assign(
        context=retrieve_context
    )
    | answer_prompt
    | chat_model
    | StrOutputParser()
)


# --------------------------------------------------
# 9. Store chat history by session ID
# --------------------------------------------------

chat_histories = {}


def get_session_history(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = (
            InMemoryChatMessageHistory()
        )

    return chat_histories[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)


# --------------------------------------------------
# 10. Test a conversation
# --------------------------------------------------

session_id = "company-policy-session-1"

config = {
    "configurable": {
        "session_id": session_id
    }
}


# First question
question_1 = (
    "How many annual leave days do employees receive?"
)

answer_1 = conversational_rag_chain.invoke(
    {"question": question_1},
    config=config,
)

print("Question 1:", question_1)
print("Answer 1:", answer_1)


# Follow-up question
question_2 = "Can I take them remotely?"

answer_2 = conversational_rag_chain.invoke(
    {"question": question_2},
    config=config,
)

print("\nQuestion 2:", question_2)
print("Answer 2:", answer_2)
