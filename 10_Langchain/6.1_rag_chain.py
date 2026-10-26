
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


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
# 3. Create a function to format retrieved documents
# --------------------------------------------------

def format_documents(documents):
    formatted_documents = []

    for document in documents:
        formatted_documents.append(
            document.page_content
        )

    return "\n\n".join(formatted_documents)


# --------------------------------------------------
# 4. Create the prompt template
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful company policy assistant.

Answer the user's question using only the
provided company policy context.

If the answer is not available in the context,
say that the company policy document does not
provide that information.

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""
)


# --------------------------------------------------
# 5. Load the local Ollama model
# --------------------------------------------------

chat_model = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)


# --------------------------------------------------
# 6. Build the RAG chain using LCEL
# --------------------------------------------------

rag_chain = (
    {
        "context": retriever | format_documents,
        "question": RunnablePassthrough(),
    }
    | prompt
    | chat_model
    | StrOutputParser()
)


# --------------------------------------------------
# 7. Ask a question
# --------------------------------------------------

question = (
    "What is the company's policy for office parking?"
)

answer = rag_chain.invoke(question)


# --------------------------------------------------
# 8. Display the answer
# --------------------------------------------------

print("Question:", question)
print("\nRAG Answer:")
print(answer)
