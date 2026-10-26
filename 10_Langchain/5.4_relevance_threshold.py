
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 1. Load the document.
file_path = (
    Path(__file__).parent
    / "documents"
    / "company_policy.txt"
)

loader = TextLoader(
    str(file_path),
    encoding="utf-8",
)

documents = loader.load()


# 2. Split the document into chunks.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=70,
    chunk_overlap=15,
    add_start_index=True,
)

chunks = splitter.split_documents(documents)


# 3. Create the embedding model.
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Create the FAISS vector store.
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model,
)


# 5. Configure retrieval.
MAX_DISTANCE = 1.0
TOP_K = 3


def retrieve_relevant_documents(question: str):
    results = vector_store.similarity_search_with_score(
        question,
        k=TOP_K,
    )

    relevant_documents = []

    for document, distance in results:
        if distance <= MAX_DISTANCE:
            relevant_documents.append((document, distance))

    return relevant_documents


# 6. Test the retriever.
question = "What is the company's policy for office parking?"

results = retrieve_relevant_documents(question)

print("Question:", question)

if len(results) == 0:
    print("\nNo sufficiently relevant documents found.")
else:
    print("\nRelevant documents:")

    for index, result in enumerate(results, start=1):
        document = result[0]
        distance = result[1]

        print(f"\n--- Result {index} ---")
        print("Content:", document.page_content)
        print("Distance:", distance)
