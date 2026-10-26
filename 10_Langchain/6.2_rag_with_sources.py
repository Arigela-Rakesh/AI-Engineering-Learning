
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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
# 3. Create the prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful company policy assistant.

Answer the question using only the provided context.

If the context does not contain the answer,
say that the company policy document does not
provide that information.

Do not invent facts or sources.

Context:
{context}

Question:
{question}

Answer:
"""
)


# --------------------------------------------------
# 4. Create the local LLM
# --------------------------------------------------

chat_model = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
)


# --------------------------------------------------
# 5. Create the answer chain
# --------------------------------------------------

answer_chain = (
    prompt
    | chat_model
    | StrOutputParser()
)


# --------------------------------------------------
# 6. Create a RAG function
# --------------------------------------------------

def answer_question(question: str):
    # Retrieve relevant documents.
    retrieved_documents = retriever.invoke(question)

    # Prepare context for the LLM.
    context_parts = []

    for document in retrieved_documents:
        context_parts.append(document.page_content)

    context = "\n\n".join(context_parts)

    # Generate the answer.
    answer = answer_chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    # Return the answer and source documents.
    return answer, retrieved_documents


# --------------------------------------------------
# 7. Ask a question
# --------------------------------------------------

question = "How many annual leave days do employees receive?"

answer, source_documents = answer_question(question)


# --------------------------------------------------
# 8. Display the answer
# --------------------------------------------------

print("Question:", question)

print("\nAnswer:")
print(answer)


# --------------------------------------------------
# 9. Display source information
# --------------------------------------------------

print("\nSources:")

for index, document in enumerate(source_documents, start=1):
    source = document.metadata.get("source", "Unknown source")
    start_index = document.metadata.get("start_index", "Unknown")

    print(f"\n--- Source {index} ---")
    print("File:", Path(source).name)
    print("Start index:", start_index)
    print("Content:", document.page_content)
